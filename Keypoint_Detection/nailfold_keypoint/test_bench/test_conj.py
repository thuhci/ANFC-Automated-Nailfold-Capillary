import argparse
import os

import cv2
import numpy as np
import yaml
from Keypoint_Detection.nailfold_keypoint.rcnn_keypoint import Keypoint
from Keypoint_Detection.nailfold_keypoint.test_bench.test_utils import (
    detect_acc, diameter_acc)
from Keypoint_Detection.nailfold_keypoint.utils.utils_visualize import \
    draw_kps_on_image


def draw_line(img, start, end):
    return cv2.line(img.copy(),[int(p) for p in start],[int(p) for p in end],(255,255,255),1)

output_dir = "test/log/conj"
debug = True
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./configs/conj.yaml", type=str, help="data file path")
args = parser.parse_args()
cfg_path = args.cfg

with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
rcnn_type = None
dataset_dir = cfg_dict.get("dataset_dir")
class_name_dict = cfg_dict.get("class_name_dict")
expname = cfg_dict.get("expname")

num_epochs = cfg_dict.get("num_epochs")
bbox_th = cfg_dict.get("bbox_th")

keypoints_classes_ids2names = cfg_dict.get("keypoints_classes_ids2names")
if keypoints_classes_ids2names is not None:
    CROSS_PADDING = cfg_dict.get("CROSS_PADDING")
    TOP_PADDING = cfg_dict.get("TOP_PADDING")
    rcnn_type = "keypoint"


if rcnn_type == "keypoint":
    model = Keypoint(cfg_dict)
    result = model.test(False)
# {"img_path": all_path,
#     "output": all_keypoints,
#     "gt": all_keypoints_gt }
all_path = result['img_path']
all_keypoints = result['output']
all_keypoints_gt= result['gt']
all_bboxes = result['bbox']

all_sensitivity = []
all_precision = []

for path, output, gt, bboxes in zip(all_path, all_keypoints, all_keypoints_gt, all_bboxes):
    print(os.path.basename(path), ":")
    output = np.array(output)
    gt = gt.numpy()[...,:2]
    # 关键点准确率
    acc = detect_acc(output, gt)
    # {"sensitivity": sensitivity,
    # "precision": precision,
    # "match_pos": np.array(l, 2)}
    match_pos = acc['match_pos']
    precision = acc['precision']
    # gt中预测正确的比例
    sensitivity = acc['sensitivity']
    print(f"sensitivity: {sensitivity.round(2)};    precision:{precision.round(2)}")
    all_sensitivity.append(sensitivity)
    all_precision.append(precision)
    if debug:
        image = cv2.imread(os.path.join(dataset_dir ,"test", "images", path))
        pred = draw_kps_on_image(image.copy(), bboxes, output, labels=[0 for _ in range(len(bboxes))])
        truth = draw_kps_on_image(image.copy(), [], gt.astype(np.int64))
        vis = np.concatenate([pred,truth], axis=0)
        h,w,_ = image.shape
        for i, pos in zip(range(match_pos.shape[0]), match_pos):
            start = gt[pos[0]][0]+np.array([0,50])
            end = output[pos[1]][0]+np.array([0,h-20])
            vis = draw_line(vis, start, end)
        cv2.imwrite(os.path.join(output_dir, path), vis)

all_sensitivity = np.array(all_sensitivity).mean()
all_precision = np.array(all_precision).mean()
print(f"===SE:{all_sensitivity.round(2)}===PR:{all_precision.round(2)}===")