import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    # x, y, w, h = box[0], box[1], box[2], box[3]
    return x, y, w, h

def parse_xml(xml_file, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = (int(root.find('size/width').text), int(root.find('size/height').text))

    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_mapping:
            continue

        class_id = class_mapping[name]

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        x, y, w, h = convert_coordinates(size, (xmin, ymin, xmax, ymax))
        labels.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return labels

def save_to_txt(txt_file, labels):
    with open(txt_file, 'w') as file:
        for label in labels:
            file.write(f"{label}\n")

if __name__ == "__main__":
    xml_folder = "data/m3fd/Annotation"
    output_folder = "data/m3fd/labels"
    class_mapping_file = "class_mapping.yml"

    # Read class mapping from YAML file
    with open(class_mapping_file, 'r') as file:
        class_mapping = yaml.safe_load(file)

    # Iterate through all XML files in the folder
    for xml_file in tqdm(os.listdir(xml_folder)):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)

            # Generate YOLO-format labels from the XML file
            labels = parse_xml(xml_path, class_mapping)

            # Create a corresponding output text file in the output folder
            txt_file = os.path.join(output_folder, os.path.splitext(xml_file)[0] + ".txt")
            save_to_txt(txt_file, labels)
