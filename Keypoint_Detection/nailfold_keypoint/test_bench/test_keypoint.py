import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from Keypoint_Detection.nailfold_keypoint.rcnn_keypoint import Keypoint
from Keypoint_Detection.nailfold_keypoint.test_bench.test_utils import (
    detect_acc, diameter_acc)
from Keypoint_Detection.nailfold_keypoint.utils.utils_visualize import \
    draw_kps_on_image


def draw_line(img, start, end):
    return cv2.line(img.copy(),[int(p) for p in start],[int(p) for p in end],(255,255,255),1)

output_dir = "test/log"
debug = True
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./configs/keypoint.yaml", type=str, help="data file path")
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

all_sensitivity = []
all_top_error = []
all_left_error = []

for path, output, gt in zip(all_path, all_keypoints, all_keypoints_gt):
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
    print(f"sensitivity: {sensitivity};    precision:{precision}")
    # 管径计算准确率
    acc = diameter_acc(output, gt, match_pos)
    # {
    #     "top_error": top_error.mean(),
    #     "left_error": left_error.mean(),
    #     "right_error": right_error.mean()
    # }
    top_error = acc['top_error']
    left_error = acc['left_error']
    right_error = acc['right_error']
    print(f"top_error: {top_error.mean()};    left_error:{left_error.mean()}    right_error:{right_error.mean()}")

    all_sensitivity.append(sensitivity)
    all_top_error += top_error.tolist()
    all_left_error += left_error.tolist()

    if debug:
        image = cv2.imread(os.path.join(dataset_dir ,"test", "images", path))
        pred = draw_kps_on_image(image.copy(), [], output)
        truth = draw_kps_on_image(image.copy(), [], gt.astype(np.int64))
        vis = np.concatenate([pred,truth], axis=0)
        h,w,_ = image.shape
        for i, pos, t_e, l_e, r_e in zip(range(match_pos.shape[0]), match_pos, top_error, left_error, right_error):
            start = gt[pos[0]][0]+np.array([0,50])
            end = output[pos[1]][0]+np.array([0,h-20])
            vis = draw_line(vis, start, end)
            p = (i+2) / (match_pos.shape[0] + 5)
            text = ((1-p) * start+ p * end).astype(np.int64)
            vis = cv2.putText(vis, f" t:{t_e.round(2)},l:{l_e.round(2)},r:{r_e.round(2)}", text, 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, path), vis)

all_sensitivity = np.array(all_sensitivity).mean()
all_top_error = np.array(all_top_error)
all_left_error = np.array(all_left_error)
print(f"===SE:{all_sensitivity}===Top:{all_top_error.mean()}===Left:{all_left_error.mean()}===")

plt.plot(np.arange(all_top_error.shape[0]), all_top_error)
plt.savefig(os.path.join(output_dir, "error.png"))