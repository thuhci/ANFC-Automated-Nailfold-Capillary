import argparse
import os

import yaml
from Keypoint_Detection.nailfold_keypoint.rcnn_keypoint import Keypoint
from Keypoint_Detection.nailfold_keypoint.rcnn_mask import Mask
from torch.utils.tensorboard import SummaryWriter

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

mask_colors = cfg_dict.get("mask_colors")
if mask_colors is not None:
    rcnn_type = "mask"


# ===================== Train ============================
os.makedirs(f"./log/{expname}/train_log",exist_ok=True)
writer = SummaryWriter(f'./log/{expname}/train_log')

if rcnn_type == "keypoint":
    model = Keypoint(cfg_dict)
    model.train(writer=writer)

elif rcnn_type == "mask":
    model = Mask(cfg_dict)
    model.train(writer=writer)