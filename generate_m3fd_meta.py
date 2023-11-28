import os
import random

# Set the path to your image folder
image_folder = 'data/m3fd/ir'

# Set the percentage split for train, test, and validation
train_percent = 60
test_percent = 20
val_percent = 20

# Get a list of all image files in the folder
all_images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Shuffle the list of images to randomize the split
random.shuffle(all_images)

# Calculate the number of images for each split
total_images = len(all_images)
train_count = int(total_images * (train_percent / 100))
test_count = int(total_images * (test_percent / 100))
val_count = total_images - train_count - test_count

# Split the list of images into train, test, and validation sets
train_set = all_images[:train_count]
test_set = all_images[train_count:train_count + test_count]
val_set = all_images[train_count + test_count:]

# Function to write file names to a text file
def write_to_file(file_path, file_list):
    with open(file_path, 'w') as file:
        for item in file_list:
            file.write("%s\n" % item)

# Write file names to the respective text files
write_to_file('data/m3fd/meta/train.txt', train_set)
write_to_file('data/m3fd/meta/test.txt', test_set)
write_to_file('data/m3fd/meta/val.txt', val_set)
