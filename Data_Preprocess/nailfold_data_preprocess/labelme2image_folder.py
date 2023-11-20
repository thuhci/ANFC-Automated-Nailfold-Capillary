"""
将segmentation分类的标注转化为血管图像分类的数据集，形如：
        root/dog/xxx.png
        root/dog/xxy.png

        root/cat/123.png
        root/cat/nsdf3.png
"""

import json
import os

import cv2
import numpy as np

from nailfold_data_preprocess.utils.my_parser import get_parse

H = 768
W = 1024
PADDING = 20

def main(args):
    raw_dir = args.input
    if args.input is None:
        raw_dir = "./data/tangshan_segmentation_clean"
    dataset_dir = args.output
    if args.output is None:
        dataset_dir = "./data/classify_mask_dataset_clean"
    output_folder = os.path.join(dataset_dir, 'train')
    test_folder = os.path.join(dataset_dir, 'test')
    imgs_path = raw_dir
    labs_path = raw_dir
    cnt = 0
    files = []
    for file in os.listdir(imgs_path):
        if file.endswith(".json"):
            continue
        files.append(file)
    files.sort()
    for file in files:
        cnt += 1
        img_path = os.path.join(imgs_path, file)
        lab_path = os.path.join(labs_path, file.replace('.png', '.json').replace('.jpg', '.json'))
        if cnt > len(files) * 3 / 4:
            output_dir = test_folder
        else:
            output_dir = output_folder
        os.makedirs(output_dir,exist_ok=True)

        with open(lab_path,"r") as f:    
            anno = json.load(f)
        img = cv2.imread(img_path)
        plys = [shape['points'] for shape in anno['shapes']]
        classes = [shape['label'] for shape in anno['shapes']]
        bboxes = []
        zero = np.zeros((img.shape), dtype=np.uint8)
        for i,ply in enumerate(plys):
            ply = np.array(ply).astype(np.int32)
            mask = cv2.fillPoly(zero.copy(), [ply], color=(255, 255, 255))
            # 计算包围盒
            xmin = max(0, ply.min(axis=0)[0] - PADDING)
            xmax = min(W, ply.max(axis=0)[0] + PADDING)
            ymin = max(0, ply.min(axis=0)[1] - PADDING)
            ymax = min(H, ply.max(axis=0)[1] + PADDING)
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            new_img = img[ymin:ymax,xmin:xmax]
            mask = mask[ymin:ymax,xmin:xmax]
            dir = os.path.join(output_dir,classes[i])
            os.makedirs(dir, exist_ok=True)
            path = os.path.join(output_dir,classes[i],file.replace(".jpg",f"_{i}.png"))
            # cv2.imwrite(path, new_img)
            cv2.imwrite(path, mask)

        print(cnt)

if __name__ == "__main__":
    args = get_parse()
    main(args)