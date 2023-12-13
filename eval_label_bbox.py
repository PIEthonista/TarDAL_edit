import cv2
import os
import yaml
from tqdm import tqdm

def read_class_mapping(yml_file):
    with open(yml_file, 'r') as stream:
        class_mapping = yaml.safe_load(stream)
    return class_mapping

def read_bbox_data(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        bbox_data = []
        for line in lines:
            line = line.strip().split()
            class_id = int(line[0])
            bbox = list(map(float, line[1:5]))
            confidence = float(line[5])
            bbox_data.append({'class_id': class_id, 'bbox': bbox, 'confidence': confidence})
            # bbox_data.append({'class_id': class_id, 'bbox': bbox})
        return bbox_data

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

def draw_bbox_on_image(image, bbox_data, class_mapping, confidence_threshold):
    for bbox_info in bbox_data:
        confidence = bbox_info['confidence']
        if confidence > confidence_threshold:
            class_id = bbox_info['class_id']
            bbox = bbox_info['bbox']

            class_name = class_mapping.get(class_id, f'Class {class_id}')
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
            text = f"{class_name} ({confidence:.2f})"
            text = f"{class_name}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return image

def process_images_in_folder(image_folder, bbox_folder, output_folder, class_mapping, confidence_threshold):
    for filename in tqdm(sorted(os.listdir(image_folder))):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Assuming image files have these extensions
            image_file_path = os.path.join(image_folder, filename)
            bbox_file_path = os.path.join(bbox_folder, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            output_image_path = os.path.join(output_folder, filename)

            if os.path.exists(bbox_file_path):
                image = cv2.imread(image_file_path)
                bbox_data = read_bbox_data(bbox_file_path)
                image_with_bbox = draw_bbox_on_image(image.copy(), bbox_data, class_mapping, confidence_threshold)
                cv2.imwrite(output_image_path, image_with_bbox)

                print(f"Processed: {filename}")

if __name__ == "__main__":    
    # infer_folder = 'experiments/tardal_ct/20231129_default/infer' # change this
    infer_folder = 'experiments/tardal_tt/20231129_default/infer' # change this
    
    image_folder_path = os.path.join(infer_folder, 'images')
    bbox_folder_path = os.path.join(infer_folder, 'labels')
    output_folder_path = os.path.join(infer_folder, 'labelled_images')
    
    confidence_threshold = 0.6 # as set in line 180, loader/m3fd.py  ->  pred_x = list(filter(lambda x: x[4] > 0.6, pred_i))
    
    # image_folder_path = '/Users/gohyixian/Desktop/twcc_results/TarDAL/tardal_ct/20231130_default_without_pretrained_fusionnet/infer/images'
    # bbox_folder_path = '/Users/gohyixian/Desktop/twcc_results/TarDAL/tardal_ct/20231130_default_without_pretrained_fusionnet/infer/labels'
    # output_folder_path = '/Users/gohyixian/Desktop/twcc_results/TarDAL/tardal_ct/20231130_default_without_pretrained_fusionnet/infer/images_labelled'
    
    # image_folder_path = '/Users/gohyixian/Desktop/twcc_results/TarDAL/tardal_ct_yolov5x/20231130/infer/images'
    # bbox_folder_path = '/Users/gohyixian/Desktop/twcc_results/TarDAL/tardal_ct_yolov5x/20231130/infer/labels'
    # output_folder_path = '/Users/gohyixian/Desktop/twcc_results/TarDAL/tardal_ct_yolov5x/20231130/infer/images_labelled'
    
    # # for all ground truth labelling
    # infer_folder = 'data/m3fd'
    # image_folder_path = os.path.join(infer_folder, 'ir')
    # bbox_folder_path = os.path.join(infer_folder, 'labels')
    # output_folder_path = os.path.join(infer_folder, 'labelled_ir')
    
    class_mapping_file = 'class_mapping.yml'

    os.makedirs(output_folder_path, exist_ok=True)

    class_mapping = read_class_mapping(class_mapping_file)

    process_images_in_folder(image_folder_path, bbox_folder_path, output_folder_path, class_mapping, confidence_threshold)

    print("DONE.")
