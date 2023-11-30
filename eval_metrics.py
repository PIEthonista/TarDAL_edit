import os
import yaml
from tqdm import tqdm

def read_class_mapping(yml_file):
    with open(yml_file, 'r') as stream:
        class_mapping = yaml.safe_load(stream)
    return class_mapping

def read_detection_results(txt_file, confidence_score=False):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        results = []
        for line in lines:
            line = line.strip().split(" ")
            class_id = int(line[0])
            bbox = list(map(float, line[1:5]))
            if confidence_score:
                confidence = float(line[5])
                results.append({'class_id': class_id, 'bbox': bbox, 'confidence': confidence})
            else:
                results.append({'class_id': class_id, 'bbox': bbox})
        return results

def calculate_iou(gt_bbox, pred_bbox):
    x1, y1, w1, h1 = gt_bbox
    x2, y2, w2, h2 = pred_bbox

    intersect_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersect_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersect_area = intersect_x * intersect_y

    area_gt = w1 * h1
    area_pred = w2 * h2
    union_area = area_gt + area_pred - intersect_area

    iou = intersect_area / max(union_area, 1e-6)
    return iou

def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = len(gt_boxes)

    for pred_box in pred_boxes:
        iou = 0.0
        
        for gt_box in gt_boxes:
            iou = calculate_iou(gt_box['bbox'], pred_box['bbox'])
            if iou >= iou_threshold and gt_box['class_id'] == pred_box['class_id']:
                true_positives += 1
                false_negatives -= 1
                break

        if iou < iou_threshold or gt_box['class_id'] != pred_box['class_id']:
            false_positives += 1

    precision = true_positives / max(true_positives + false_positives, 1e-6)
    recall = true_positives / max(true_positives + false_negatives, 1e-6)

    return precision, recall

def calculate_ap(precision, recall):
    return precision * recall / max(precision + recall, 1e-6)

def calculate_map(ap_list):
    return sum(ap_list) / max(len(ap_list), 1e-6)

if __name__ == "__main__":
    # Replace these paths with the actual paths of your ground truth, predicted result files, and class mapping file
    gt_folder_path = 'data/m3fd/labels' # fixed
    # pred_folder_path = 'experiments/tardal_ct/20231129_default/infer/labels' # change this
    pred_folder_path = 'experiments/tardal_tt/20231129_default/infer/labels' # change this
    class_mapping_file = 'class_mapping.yml' # fixed
    
    confidence_threshold = 0.6 # as set in line 180, loader/m3fd.py  ->  pred_x = list(filter(lambda x: x[4] > 0.6, pred_i))

    class_mapping = read_class_mapping(class_mapping_file)
    num_classes = len(class_mapping)

    ap_list = []
    ap50_list = []
    
    pred_folder, _ = os.path.split(pred_folder_path)

    with open(os.path.join(pred_folder, "eval_metrics.txt"), "w") as results_file:
        for filename in tqdm(os.listdir(pred_folder_path)):
            if filename.endswith(".txt"):
                gt_file_path = os.path.join(gt_folder_path, filename)
                pred_file_path = os.path.join(pred_folder_path, filename)

                gt_results = read_detection_results(gt_file_path, confidence_score=False)
                pred_results = read_detection_results(pred_file_path, confidence_score=True)

                class_aps = []
                class_aps50 = []

                results_file.write(f"Results for {filename}:\n")

                for class_id in range(num_classes):
                    gt_boxes = [box for box in gt_results if box['class_id'] == class_id]
                    pred_boxes = [box for box in pred_results if box['class_id'] == class_id and box['confidence'] > confidence_threshold]

                    precision, recall = calculate_precision_recall(gt_boxes, pred_boxes)
                    ap = calculate_ap(precision, recall)
                    class_aps.append(ap)

                    precision50, _ = calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5)
                    ap50 = calculate_ap(precision50, 1.0)
                    class_aps50.append(ap50)

                    results_file.write(f"  Class {class_id}: AP={ap:.4f}, AP50={ap50:.4f}\n")

                mAP = calculate_map(class_aps)
                ap_list.append(mAP)

                mAP50 = calculate_map(class_aps50)
                ap50_list.append(mAP50)

                results_file.write(f"{filename}  mAP: {mAP:.4f}, mAP50: {mAP50:.4f}\n\n")

        overall_mAP = calculate_map(ap_list)
        overall_mAP50 = calculate_map(ap50_list)

        results_file.write(f"Overall mAP: {overall_mAP:.4f}, Overall mAP50: {overall_mAP50:.4f}\n")
