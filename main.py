import sys
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QProgressBar, QSplitter, QFrame, QSpacerItem, QSizePolicy, QFileDialog
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import mediapipe as mp

class RayTracer(QThread):
    update_progress = pyqtSignal(int)
    elapsed_time = pyqtSignal(str)
    estimated_time = pyqtSignal(str)
    pixels_per_second = pyqtSignal(str)
    image_output = pyqtSignal(np.ndarray)
    finished = pyqtSignal(bool)

    def __init__(self, values, octree, mat_table):
        super().__init__()
        self.values = values
        self.octree = octree
        self.done = False
        self.mat_table = mat_table
    
    def stop(self):
        self.done = True

    def toBinaryStr(self, value, bit_depth):
        if value < 0:
            value = (1 << bit_depth) + value
        binary_str = format(value, '0{}b'.format(bit_depth))
        return binary_str[-bit_depth:]
    
    def sanitiseNode(self, node):
        if(isinstance(node, int)):
            return node
        elif(isinstance(node, str)):
            if '/' in node:
                node = node[:-2]
            return int(node, 16)
        else:
            raise TypeError(f"Unsupported input type: {type(node)}")
        
    def parseMatTable(self, mat_table):
        material_table = [list(map(int, item.split())) for item in mat_table]
        return material_table

    def withinAABB(self, position, aabb_min, aabb_max):
        return (position[0] >= aabb_min[0] and position[1] >= aabb_min[1] and position[2] >= aabb_min[2] and
                position[0] <= aabb_max[0] and position[1] <= aabb_max[1] and position[2] <= aabb_max[2])

    def justOutsideAABB(self, position, aabb_min, aabb_max):
        outside_x = (position[0] == aabb_min[0] - 1 or position[0] == aabb_max[0] + 1)
        outside_y = (position[1] == aabb_min[1] - 1 or position[1] == aabb_max[1] + 1)
        outside_z = (position[2] == aabb_min[2] - 1 or position[2] == aabb_max[2] + 1)

        within_x = (position[0] >= aabb_min[0] and position[0] <= aabb_max[0])
        within_y = (position[1] >= aabb_min[1] and position[1] <= aabb_max[1])
        within_z = (position[2] >= aabb_min[2] and position[2] <= aabb_max[2])

        return (outside_x and within_y and within_z) or (within_x and outside_y and within_z) or (
                           within_x and within_y and outside_z) or (outside_x and outside_y and within_z) or (
                           outside_x and within_y and outside_z) or (within_x and outside_y and outside_z) or (
                           outside_x and outside_y and outside_z)

    def traverseTree(self, ray_pos, root, oct_size, aabb_min, aabb_max):
        depth = 0
        node = self.sanitiseNode(root[0])
        x_bin = self.toBinaryStr(ray_pos[0], self.coord_bit_length)
        y_bin = self.toBinaryStr(ray_pos[1], self.coord_bit_length)
        z_bin = self.toBinaryStr(ray_pos[2], self.coord_bit_length)

        while (node<2**31 and depth < self.coord_bit_length):
            octant = int(('0b' + z_bin[depth] + y_bin[depth] + x_bin[depth]), base=0)
            depth += 1
            oct_size /= 2

            aabb_min = (aabb_min + oct_size * (np.array([x_bin[depth - 1], y_bin[depth - 1], z_bin[depth - 1]], dtype=int))).astype(int)
            aabb_max = (aabb_min + np.array([oct_size - 1, oct_size - 1, oct_size - 1], dtype=int)).astype(int)

            node = self.sanitiseNode(root[octant + node])

        return node, oct_size, aabb_min, aabb_max

    def stepRay(self, ray_pos, ray_dir, oct_size, aabb_min, aabb_max):
        temp_ray_dir = np.copy(ray_dir)
        while (temp_ray_dir[0] * temp_ray_dir[0] + temp_ray_dir[1] * temp_ray_dir[1] + temp_ray_dir[2] * temp_ray_dir[2] < oct_size * oct_size):
            temp_ray_dir *= 2
        while not self.justOutsideAABB(ray_pos, aabb_min, aabb_max):
            temp_position = ray_pos + temp_ray_dir
            if (self.withinAABB(temp_position, aabb_min, aabb_max) or self.justOutsideAABB(temp_position, aabb_min, aabb_max)):
                ray_pos = temp_position
            temp_ray_dir[0] = int(temp_ray_dir[0] / 2) if (abs(temp_ray_dir[0]) > 1) else (1 if (temp_ray_dir[0] >= 0) else -1)
            temp_ray_dir[1] = int(temp_ray_dir[1] / 2) if (abs(temp_ray_dir[1]) > 1) else (1 if (temp_ray_dir[1] >= 0) else -1)
            temp_ray_dir[2] = int(temp_ray_dir[2] / 2) if (abs(temp_ray_dir[2]) > 1) else (1 if (temp_ray_dir[2] >= 0) else -1)
        return ray_pos

    def applyGammaCorrection(self, colour, gamma=2.2):
        return np.clip(255 * (colour / 255) ** (1 / gamma), 0, 255).astype(np.uint8)

    def run(self):
        im_height = self.values[0]
        im_width = self.values[1]
        self.coord_bit_length = self.values[2]
        cam_pos = np.array(self.values[3:6], dtype=int)
        cam_norm = np.array(self.values[6:9], dtype=int)
        cam_up = np.array(self.values[9:12], dtype=int)
        cam_right = np.array(self.values[12:15], dtype=int)

        # World data
        octree = self.octree if self.octree is not None else [
            1, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            9, 
            0xFFFFFF02, 
            0xFFFFFF03, 
            0xFFFFFF01, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            17, 
            0xFFFFFF02, 
            0xFFFFFF03, 
            0xFFFFFF01, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF00, 
            0xFFFFFF02, 
            0xFFFFFF03, 
            0xFFFFFF01
            ]
        material_table = self.parseMatTable(self.mat_table) if self.mat_table is not None else [
            [0, 0, 0], 
            [255, 255, 255], 
            [0, 255, 0], 
            [0, 0, 255], 
            [255, 0, 0]
            ]
        
        # Image placeholder
        image = np.zeros((im_height, im_width, 3), dtype=np.uint8)

        total_pixels = im_height * im_width
        progress = 0

        start_time = time.time()

        for y in range(im_height):
            if self.done:
                break
            for x in range(im_width):
                centered_x = int(x - (im_width / 2))
                centered_y = int((im_height / 2) - y)
                ray_dir = cam_norm + cam_right * centered_x + cam_up * centered_y

                ray_pos = np.copy(cam_pos)

                world_size = 2 ** self.coord_bit_length
                world_min = np.array([0, 0, 0], dtype=int)
                world_max = np.array([world_size - 1, world_size - 1, world_size - 1], dtype=int)

                oct_size = world_size
                aabb_min = world_min
                aabb_max = world_max
                root = octree

                while self.withinAABB(ray_pos, world_min, world_max):

                    mid, oct_size, aabb_min, aabb_max = self.traverseTree(ray_pos, root, world_size, world_min, world_max)
                    mid = mid & 0xff

                    if mid == 0:
                        ray_pos = self.stepRay(ray_pos, ray_dir, oct_size, aabb_min, aabb_max)

                    if mid > 0:
                        mid = mid % len(material_table)
                        hit_normal = np.array([0, 0, 0])
                        if ray_pos[0] == aabb_min[0]:
                            hit_normal = np.array([-1, 0, 0])
                        elif ray_pos[0] == aabb_max[0]:
                            hit_normal = np.array([1, 0, 0])
                        elif ray_pos[1] == aabb_min[1]:
                            hit_normal = np.array([0, -1, 0])
                        elif ray_pos[1] == aabb_max[1]:
                            hit_normal = np.array([0, 1, 0])
                        elif ray_pos[2] == aabb_min[2]:
                            hit_normal = np.array([0, 0, -1])
                        elif ray_pos[2] == aabb_max[2]:
                            hit_normal = np.array([0, 0, 1])

                        light_dir = cam_pos - ray_pos
                        light_dir = light_dir / np.linalg.norm(light_dir)

                        brightness_factor = (np.dot(light_dir, hit_normal)) ** 2

                        colour = np.array(material_table[mid]) * brightness_factor
                        colour = np.clip(colour, 0, 255).astype(np.uint8)
                        colour = self.applyGammaCorrection(colour)

                        break
                else:
                    colour = [0, 0, 0]

                image[y, x] = colour

                progress += 1
                percent_complete = int(progress / total_pixels * 100)
                self.update_progress.emit(percent_complete)

                # Calculate estimated time and pixels per second
                time_taken = time.time() - start_time
                pixels_per_second = progress / time_taken if time_taken > 0 else 0
                remaining_time = (total_pixels - progress) / pixels_per_second if pixels_per_second > 0 else 0

                self.elapsed_time.emit(f"Elapsed time: {time_taken:.1f} s")
                self.estimated_time.emit(f"Estimated time left: {remaining_time:.1f} s")
                self.pixels_per_second.emit(f"Pixels/s: {pixels_per_second:.1f}")
                self.image_output.emit(image)

        self.done = True
        self.image_output.emit(image)
        self.finished.emit(True)

class MainWindow(QWidget):
    rendering = False 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPGA Ray Tracer Model")
        self.setGeometry(150, 100, 1100, 650)
        self.initUI()

    def initUI(self):
        self.octree = None
        self.mat_table = None
        main_layout = QHBoxLayout()
        self.setStyleSheet("background-color: #444444;")

        # Left panel
        self.left_panel = QLabel()
        self.left_panel.setStyleSheet("background-color: lightgray; border-radius: 2px;")
        self.left_panel.setFrameShape(QFrame.StyledPanel)
        self.left_panel.setAlignment(Qt.AlignCenter)

        # Right panel
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #555555; color: white; border-radius: 8px;")
        right_panel.setFrameShape(QFrame.StyledPanel)

        # Parameters layout
        parameters_layout = QVBoxLayout()
        parameters_layout.setContentsMargins(20, 20, 20, 20)

        # Title for right panel
        title_label = QLabel("Parameters")
        title_label.setFont(QFont("Helvetica", 22))
        title_label.setStyleSheet("color: white;")

        parameters_layout.addWidget(title_label)

        parameters_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        helvetica_font = QFont("Helvetica", 14)

        labels = [
            "Image Height:",
            "Image Width:",
            "World Bit Depth:",
            "Camera Position (x, y, z):",
            "Camera Direction (x, y, z):",
            "Camera Up (x, y, z):",
            "Camera Right (x, y, z):"
        ]
        default_values = ["300", "300", "10", "200,200,0", "0,0,100", "0,1,0", "1,0,0"]
        

        # Input boxes
        self.input_boxes = []
        for label_text, default_value in zip(labels, default_values):
            label = QLabel(label_text)
            label.setFont(helvetica_font)
            input_box = QLineEdit()
            input_box.setFixedSize(150, 30)
            input_box.setFont(helvetica_font)
            input_box.setText(default_value)
            input_box.setStyleSheet("background-color: #333333; color: white; border: 1px solid gray; border-radius: 5px;")
            self.input_boxes.append(input_box)

            hbox = QHBoxLayout()
            hbox.addWidget(label)
            hbox.addWidget(input_box)
            parameters_layout.addLayout(hbox)

        # Octee file select button
        oct_file_select_button = QPushButton('Select Octree File')
        oct_file_select_button.clicked.connect(self.selectOctFile)
        oct_file_select_button.setFont(helvetica_font)
        oct_file_select_button.setFixedWidth(150)
        oct_file_select_button.setStyleSheet("QPushButton {background-color: gray; color: white; border-radius: 5px; min-height: 20px;}"
                                    "QPushButton:pressed {background-color: darkGray;}")

        # Material table file select button
        mat_file_select_button = QPushButton('Select Material File')
        mat_file_select_button.clicked.connect(self.selectMatFile)
        mat_file_select_button.setFont(helvetica_font)
        mat_file_select_button.setFixedWidth(150)
        mat_file_select_button.setStyleSheet("QPushButton {background-color: gray; color: white; border-radius: 5px; min-height: 20px;}"
                                    "QPushButton:pressed {background-color: darkGray;}")

        bbox = QHBoxLayout()
        bbox.addWidget(oct_file_select_button)
        bbox.addWidget(mat_file_select_button)
        parameters_layout.addLayout(bbox)


        # Open camera button 
        self.cam_button = QPushButton("Open gesture control")
        self.cam_button.clicked.connect(self.openCam)
        self.cam_button.setFont(helvetica_font)
        self.cam_button.setStyleSheet("QPushButton {background-color: #3498db; color: white; border-radius: 5px; min-height: 30px;}"
                                    "QPushButton:pressed {background-color: #2980b9;}"
                                    "QPushButton:disabled {background-color: #2c3e50;  color: #bdc3c7;}")
        parameters_layout.addWidget(self.cam_button)
        self.cam_button.setEnabled(False)

        # Error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: rgba(255, 0, 0, 150);")
        self.error_label.setFont(helvetica_font)
        parameters_layout.addWidget(self.error_label)

        parameters_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))


        # Elapsed time label
        self.elapsed_time_label = QLabel()
        self.elapsed_time_label.setStyleSheet("color: white;")
        self.elapsed_time_label.setFont(helvetica_font)
        parameters_layout.addWidget(self.elapsed_time_label)

        # Estimated time label
        self.estimated_time_label = QLabel()
        self.estimated_time_label.setStyleSheet("color: white;")
        self.estimated_time_label.setFont(helvetica_font)
        parameters_layout.addWidget(self.estimated_time_label)

        # Pixels per second label
        self.pixels_per_second_label = QLabel()
        self.pixels_per_second_label.setStyleSheet("color: white;")
        self.pixels_per_second_label.setFont(helvetica_font)
        parameters_layout.addWidget(self.pixels_per_second_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        parameters_layout.addWidget(self.progress_bar)

        # Submit button 
        self.submit_button = QPushButton("Render")
        self.submit_button.clicked.connect(self.submit)
        self.submit_button.setFont(helvetica_font)
        self.submit_button.setStyleSheet("QPushButton {background-color: #3498db; color: white; border-radius: 5px; min-height: 40px;}"
                                    "QPushButton:pressed {background-color: #2980b9;}")
        parameters_layout.addWidget(self.submit_button)

        right_panel.setLayout(parameters_layout)

        # Splitter
        splitter = QSplitter()
        splitter.addWidget(self.left_panel)
        splitter.addWidget(right_panel)
        splitter.setCollapsible(0, False) 
        initial_width = int(self.width() * 0.7)
        splitter.setSizes([initial_width, self.width() - initial_width])

        main_layout.addWidget(splitter)

        self.setLayout(main_layout)

    def selectOctFile(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Octree File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                self.octree = file.readlines()
    
    def selectMatFile(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Material Table File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                self.mat_table = file.readlines()

    def submit(self):
        if not self.rendering:
            empty_fields = [field for field in self.input_boxes if not field.text()]
            if empty_fields:
                self.error_label.setText("Please fill in all the fields.")
            else:
                if self.validateInput():
                    self.cam_button.setEnabled(False)
                    self.error_label.clear()
                    self.progress_bar.setValue(0)
                    self.elapsed_time_label.clear()
                    self.estimated_time_label.clear()
                    self.pixels_per_second_label.clear()
                    values = self.getValues()
                    octree = self.octree
                    mat_table = self.mat_table

                    self.ray_tracer = RayTracer(values, octree, mat_table)
                    self.ray_tracer.update_progress.connect(self.progress_bar.setValue)
                    self.ray_tracer.elapsed_time.connect(self.elapsed_time_label.setText)
                    self.ray_tracer.estimated_time.connect(self.estimated_time_label.setText)
                    self.ray_tracer.pixels_per_second.connect(self.pixels_per_second_label.setText)
                    self.ray_tracer.image_output.connect(self.displayImage)
                    self.ray_tracer.finished.connect(self.finished)
                    self.ray_tracer.start()
                    self.rendering = True
                    self.submit_button.setText("Stop Rendering")
                else:
                    self.error_label.setText("Please enter valid data.")
        else:
          if hasattr(self, 'ray_tracer'):
              self.ray_tracer.stop()

    def validateInput(self):
        for i, input_box in enumerate(self.input_boxes):
            if i == 0 or i == 1 or i == 2:
                try:
                    int(input_box.text())
                except ValueError:
                    return False
            elif i == 3 or i == 4 or i == 5 or i == 6:
                data = input_box.text().replace(" ", "").split(",")
                if len(data) != 3:
                    return False
                for d in data:
                    try:
                        int(d)
                    except ValueError:
                        return False
            else:
                return False
        return True

    def getValues(self):
        values = []
        for i, input_box in enumerate(self.input_boxes):
            if i == 0 or i == 1 or i == 2:
                values.append(int(input_box.text()))
            elif i == 3 or i == 4 or i == 5 or i == 6:
                data = input_box.text().replace(" ", "").split(",")
                values.extend(int(d) for d in data)
        return values

    def displayImage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.left_panel.setPixmap(pixmap.scaled(self.left_panel.size(), Qt.KeepAspectRatio))

    def finished(self, done):
        self.rendering = False
        self.submit_button.setText("Render")
        self.cam_button.setEnabled(True)
    
    def openCam(self):
        empty_fields = [field for field in self.input_boxes if not field.text()]
        if empty_fields:
            self.error_label.setText("Please fill in all the fields.")
        else:
            if self.validateInput():
                values = self.getValues()
                coord_bit_length = values[2]
                cam_pos = np.array([values[3], values[4], values[5]], dtype=int)
                cam_norm = np.array([values[6], values[7], values[8]], dtype=int)
                cam_up = np.array([values[9], values[10], values[11]], dtype=int)
                cam_right = np.array([values[12], values[13], values[14]], dtype=int)
                cap = cv2.VideoCapture(0)

                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

                mp_drawing = mp.solutions.drawing_utils

                position_saved = False
                origin = None
                call_render = False

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks:
                        # for hand_landmarks in results.multi_hand_landmarks:
                            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        hand_landmarks = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        finger_count = self.countFingers(hand_landmarks)
                        current_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        if self.twoFrontFingers(hand_landmarks): 
                            if not position_saved:
                                origin = current_pos
                                position_saved = True 
                            elif origin != None: 
                                dx =  int((origin.x - current_pos.x)*-50)
                                dy = int((origin.y - current_pos.y)*-50)

                                cam_pos = np.clip((cam_pos + dx*cam_right + dy*cam_up), 0, (2**coord_bit_length)-1)
                                new_pos = str(cam_pos[0])+", "+str(cam_pos[1])+", "+str(cam_pos[2])
                                self.input_boxes[3].setText(new_pos)

                                cv2.line(frame, 
                                         (int(origin.x * frame.shape[1]), 
                                          int(origin.y * frame.shape[0])), 
                                          (int(current_pos.x * frame.shape[1]), 
                                           int(current_pos.y * frame.shape[0])), 
                                           (0, 255, 0), 2)
                        elif finger_count == 5: 
                            position_saved = False 
                            origin = None
                            call_render = True
                            break
                        else: 
                            position_saved = False 
                            origin = None

                    cv2.imshow('Hand Gesture', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
                if call_render:
                    print("do the thing here")
                    self.submit()
                
            else:
                self.error_label.setText("Please enter valid data.")

    def countFingers(self, lst):
        count = 0
        thresh = (lst.landmark[0].y - lst.landmark[9].y)*50

        if((lst.landmark[5].y - lst.landmark[8].y)*100 > thresh):
            count += 1
        if((lst.landmark[9].y - lst.landmark[12].y)*100 > thresh):
            count += 1
        if((lst.landmark[13].y - lst.landmark[16].y)*100 > thresh):
            count += 1
        if((lst.landmark[17].y - lst.landmark[20].y)*100 > thresh):
            count += 1
        if((lst.landmark[5].x - lst.landmark[4].x)*100 > 5 and lst.landmark[9].x > lst.landmark[5].x) or \
            ((lst.landmark[4].x - lst.landmark[5].x)*100 > 5 and lst.landmark[9].x < lst.landmark[5].x):
            count += 1
        return count 

    def twoFrontFingers(self, lst):
        mp_hands = mp.solutions.hands
        thumb_tip = lst.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = lst.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = lst.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = lst.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = lst.landmark[mp_hands.HandLandmark.PINKY_TIP]

        thumb_y = thumb_tip.y
        index_y = index_tip.y
        middle_y = middle_tip.y
        ring_y = ring_tip.y
        pinky_y = pinky_tip.y

        if index_y < lst.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and \
        middle_y < lst.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and \
        ring_y > lst.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and \
        pinky_y > lst.landmark[mp_hands.HandLandmark.PINKY_DIP].y:
            return True
        return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
