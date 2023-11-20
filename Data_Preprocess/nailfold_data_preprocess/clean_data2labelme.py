"""
将图像分类的数据集清洗过后，修正原本labelme标注的原数据
"""
import json
import os
import re

import cv2
import numpy as np

from nailfold_data_preprocess.utils.my_parser import get_parse

H = 768
W = 1024
PADDING = 20

def clean2labelme(args):
    """
    图像分类经过数据清洗后，重新编辑原数据集的json标注
    """
    dataset_dir = args.input
    if args.input is None:
        dataset_dir = "./data/classify_dataset"
    raw_dir = args.output
    if args.output is None:
        raw_dir = "./data/tangshan_segmentation"
    dataset = os.path.join(dataset_dir, 'train')
    class_list = os.listdir(dataset) 
    for classes in class_list:
        train_dataset = os.path.join(dataset_dir, 'train', classes)
        test_dataset = os.path.join(dataset_dir, 'test', classes)
        for file in os.listdir(train_dataset) + os.listdir(test_dataset):
            print(file)
            match = re.match("([0-9]_[0-9]*_[0-9]*)_([0-9]*).png",file)
            if match is None:
                continue
            file_name = match.group(1)
            id = int(match.group(2))

            with open(os.path.join(raw_dir, file_name+".json"), "r") as f:
                anno = json.load(f)
            # 添加valid域, 确认label
            anno['shapes'][id]['valid'] = 1
            anno['shapes'][id]['label'] = classes
            with open(os.path.join(raw_dir, file_name+".json"), "w") as f:
                json.dump(anno, f, indent=4)


def clear_invalid(args):
    """
    去除清洗掉的数据标注
    """
    raw_dir = args.input
    if args.input is None:
        raw_dir = "./data/tangshan_segmentation"
    new_dir = args.output
    if args.output is None:
        new_dir = "./data/tangshan_segmentation_clean"
    file_list = os.listdir(raw_dir) 

    os.makedirs(new_dir, exist_ok=True)
    for file in file_list:
        if not file.endswith("json"):
            continue
        with open(os.path.join(raw_dir, file), "r") as f:
            anno = json.load(f)
        # 添加valid域, 确认label
        valid = []
        for shape in anno['shapes']:
            if 'valid' in shape:
                valid.append(shape)
        anno['shapes'] = valid
        if len(valid) == 0:
            continue
        jpg_path = os.path.join(raw_dir, file.replace(".json", ".jpg"))
        new_jpg_path = os.path.join(new_dir, file.replace(".json", ".jpg"))
        os.system(f"cp {jpg_path} {new_jpg_path}")
        with open(os.path.join(new_dir, file), "w") as f:
            json.dump(anno, f, indent=4)



if __name__ == "__main__":
    args = get_parse()
    clean2labelme(args)
    clear_invalid(args)