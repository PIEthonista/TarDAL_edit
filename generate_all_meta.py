import os
from tqdm import tqdm

folder_path = '/work/u5832291/yixian/TarDAL_edit/data/m3fd/ir'
output_file = '/work/u5832291/yixian/TarDAL_edit/data/m3fd/meta/train.txt'

# Get a list of all PNG files in the specified folder
png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Sort the list of filenames
png_files.sort()

# Write the filenames to a txt file
with open(output_file, 'w') as file:
    for filename in tqdm(png_files):
        file.write(filename + '\n')

print(f"File names have been saved to {output_file}.")
