
import json
import os

def txt_to_json(txt_folder, json_output_path, gt=True):
    data = {"annotations": []}

    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(txt_folder, filename), 'r') as file:
                lines = file.readlines()

                for line in lines:
                    values = line.strip().split()
                    if len(values) == 6:
                        category, x, y, width, height, confidence = map(float, values)
                        annotation = {
                            "image_id": int(os.path.splitext(filename)[0]),
                            "category_id": int(category),
                            "bbox": [x, y, width, height],
                            "score": confidence
                        }
                        data["annotations"].append(annotation)

    with open(json_output_path, 'w') as json_file:
        json.dump(data, json_file)

# Replace 'txt_folder' with the path to your folder containing txt files
# Replace 'json_output_path' with the desired path for the output JSON file
txt_folder = '/path/to/txt/files'
json_output_path = '/path/to/output/json/file.json'

txt_to_json(txt_folder, json_output_path)



from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno = COCO(anno_json)  # init annotations api
pred = anno.loadRes(pred_json)  # init predictions api
eval = COCOeval(anno, pred, 'bbox')
if is_coco:
    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)