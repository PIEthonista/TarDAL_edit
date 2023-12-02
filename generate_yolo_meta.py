input_file_path = '/work/u5832291/yixian/TarDAL_edit/data/m3fd/meta/val.txt'
output_file_path = '/work/u5832291/huiern/m3fd/visible/val.txt'
custom_text = './images/'

with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()

# Prepend custom text to each line
modified_lines = [f'{custom_text}{line.strip()}\n' for line in lines]

with open(output_file_path, 'w') as output_file:
    output_file.writelines(modified_lines)

print(f'Content has been successfully written to {output_file_path}')
