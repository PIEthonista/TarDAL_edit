import cv2
import os
import yaml
from tqdm import tqdm
from pathlib import Path
from config import from_dict
import torch

import loader
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

def draw_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x, y, w, h = bbox[2:]
    class_id = bbox[1]
    x *= width
    y *= height
    w *= width
    h *= height

    color = get_class_color(int(class_id))

    # Calculate bounding box coordinates
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2

    # Draw bounding box on the image
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return image

def read_class_mapping(yml_file):
    with open(yml_file, 'r') as stream:
        class_mapping = yaml.safe_load(stream)
    return class_mapping


def get_class_color(class_id):
    # Assign colors for each class
    class_colors = {
        0: (0, 0, 255),    # People (Red)
        1: (0, 255, 0),    # Car (Green)
        2: (255, 0, 0),    # Bus (Blue)
        3: (255, 255, 0),  # Lamp (Yellow)
        4: (255, 0, 255),  # Motorcycle (Magenta)
        5: (0, 255, 255)   # Truck (Cyan)
    }
    return class_colors.get(class_id, (255, 255, 255))  # Default to white if class not found


def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

def draw_bbox_on_image(image, bbox_data, class_mapping):
    for bbox_info in bbox_data:
        class_id = bbox_info['class_id']
        bbox = bbox_info['bbox']
        # confidence = bbox_info['confidence']

        # class_name = class_mapping.get(class_id, f'Class {class_id}')
        class_name = f"{class_id} " + get_key_by_value(class_mapping, class_id)

        color = get_class_color(class_id)
        thickness = 2
        font_scale = 0.5
        font_thickness = 1

        # Convert bbox from relative to absolute coordinates
        h, w, _ = image.shape
        x, y, bw, bh = bbox
        x1, y1, x2, y2 = int(w * (x - bw / 2)), int(h * (y - bh / 2)), int(w * (x + bw / 2)), int(h * (y + bh / 2))

        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Display class name and confidence
        # text = f"{class_name} ({confidence:.2f})"
        text = f"{class_name}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return image

# def process_images_in_folder(image_folder, bbox_folder, output_folder, class_mapping):
#     for filename in tqdm(os.listdir(image_folder)):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):  # Assuming image files have these extensions
#             image_file_path = os.path.join(image_folder, filename)
#             bbox_file_path = os.path.join(bbox_folder, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
#             output_image_path = os.path.join(output_folder, filename)

#             if os.path.exists(bbox_file_path):
#                 image = cv2.imread(image_file_path)
#                 bbox_data = read_bbox_data(bbox_file_path)
#                 image_with_bbox = draw_bbox_on_image(image.copy(), bbox_data, class_mapping)
#                 cv2.imwrite(output_image_path, image_with_bbox)

#                 print(f"Processed: {filename}")

if __name__ == "__main__":
    
    # init config
    config = yaml.safe_load(Path('config/official/train/tardal-ct.yaml').open('r'))
    config = from_dict(config)  # convert dict to object
    
    # load dataset
    data_t = getattr(loader, config.dataset.name)
    t_dataset = data_t(root=config.dataset.root, mode='train', config=config)
    
    sample = t_dataset[500]
    
    # sample = {
    #     'name': name,
    #     'ir': ir, 'vi': vi,
    #     'ir_w': ir_w, 'vi_w': vi_w, 'mask': mask, 'cbcr': cbcr,
    #     'labels': labels_o
    # }
    
    print(sample['name'])
    print(sample['ir'].shape)
    print(sample['vi'].shape)
    print(torch.min(sample['vi']), torch.max(sample['vi']))
    print(sample['ir_w'].shape)
    print(sample['vi_w'].shape)
    print(sample['labels'])
    
    image = sample['vi']
    image_pil = TF.to_pil_image(image.squeeze())
    
    # Save labeled images
    for i in range(len(sample['labels'])):
        # Convert PyTorch tensor to PIL Image
    
        # Draw bounding box on the image
        image_pil = draw_bbox(image_pil.copy(), sample['labels'][i])

    # Save the labeled image
    image_pil.save(f"labeled_image.png")
    
    
    

    # # infer_folder = 'experiments/tardal_ct/20231129_default/infer' # change this
    # infer_folder = 'experiments/tardal_tt/20231129_default/infer' # change this
    
    # image_folder_path = os.path.join(infer_folder, 'images')
    # bbox_folder_path = os.path.join(infer_folder, 'labels')
    # output_folder_path = os.path.join(infer_folder, 'labelled_images')
    
    # # # for all ground truth labelling
    # # infer_folder = 'data/m3fd'
    # # image_folder_path = os.path.join(infer_folder, 'ir')
    # # bbox_folder_path = os.path.join(infer_folder, 'labels')
    # # output_folder_path = os.path.join(infer_folder, 'labelled_ir')
    
    # class_mapping_file = 'class_mapping.yml'

    # os.makedirs(output_folder_path, exist_ok=True)

    # class_mapping = read_class_mapping(class_mapping_file)

    # process_images_in_folder(image_folder_path, bbox_folder_path, output_folder_path, class_mapping)

    # print("DONE.")
