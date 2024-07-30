import re 

input_file = 'octree_output.txt'
intermediate_file = 'intermediate-octree_output.txt'
final_output_file = 'imperial.txt'
mat_file = 'imperial_mat.txt'


def to_hex_32bit(value):
    return f"{value:08x}"

def add_line_numbers(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for i, line in enumerate(lines):
            file.write(f"{i}: {line}")

def create_node_mapping(file_with_lines):
    node_to_line_index = {}
    with open(file_with_lines, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        parts = line.split(':')
        node_number = int(parts[1].split()[1])
        node_to_line_index[node_number] = i
    
    return node_to_line_index, lines

def extract_rgb(line_content):
    match = re.search(r'\((\d+), (\d+), (\d+)\)', line_content)
    if match:
        return tuple(map(int, match.groups()))
    return None

def update_and_format_lines(node_to_line_index, lines, output_file, mat_output_file):
    color_to_tag = {}
    next_tag = 1

    with open(output_file, 'w') as file, open(mat_output_file, 'w') as mat_file:
        for index, line_content in enumerate(lines):
            if "First Child Index" in line_content:
                parts = line_content.split('First Child Index = ')
                first_child_index = int(parts[1].strip())
                if first_child_index in node_to_line_index:
                    hex_index = to_hex_32bit(node_to_line_index[first_child_index])
                    updated_line = f"{hex_index}\n"
                else:
                    updated_line = line_content
            else:
                rgb = extract_rgb(line_content)
                if rgb:
                    if rgb not in color_to_tag:
                        tag = f"FFFFFF{next_tag:02x}"
                        color_to_tag[rgb] = tag
                        mat_file.write(f"{rgb[0]} {rgb[1]} {rgb[2]}\n")
                        next_tag += 1
                    updated_line = f"{color_to_tag[rgb]}\n"
                else:
                    updated_line = f"FFFFFF00\n"  # Default tag for nodes without material
            file.write(updated_line)

def main():
    add_line_numbers(input_file, intermediate_file)
    node_to_line_index, lines = create_node_mapping(intermediate_file)
    update_and_format_lines(node_to_line_index, lines, final_output_file, mat_file)
    print(f"Processed {len(lines)} lines and saved to {final_output_file} and {mat_file}")

if __name__ == "__main__":
    main()
