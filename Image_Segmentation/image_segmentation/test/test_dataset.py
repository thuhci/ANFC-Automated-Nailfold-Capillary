import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


class SegmentDataset(Dataset):
    '''
    Instance Segmentation Dataset
    '''
    def __init__(self, root, classes_list):
        self.root = root
        self.classes_list = classes_list
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        # self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
        self.annotations_files = [x.replace('images', 'annotations').replace('.png', '.json').replace('.jpg', '.json') for x in self.imgs_files]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(
            self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        h,w,_ = img_original.shape
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        with open(annotations_path) as f:
            data = json.load(f)
            # 所有多边形
            plys = [shape['points'] for shape in data['shapes']]
            labels = [shape['label'] for shape in data['shapes']]
            mask = np.zeros((h,w), dtype=np.uint8)
            for ply, label in zip(plys, labels):
                if label not in self.classes_list:
                    continue
                # 多边形转mask
                ply = np.array(ply).astype(np.int32)
                mask = cv2.fillPoly(mask.copy(), [ply], 1)

        target = {}
        target["image_id"] = torch.tensor([idx])
        img = F.to_tensor(img_original)
        mask = torch.as_tensor(mask, dtype=torch.bool)
            
        return img_path, img, mask

    def __len__(self):
        return len(self.imgs_files)
