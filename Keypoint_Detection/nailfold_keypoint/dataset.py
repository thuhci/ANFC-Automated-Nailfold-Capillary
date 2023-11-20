import json
import os
import random

import albumentations as A  # Library for augmentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from Keypoint_Detection.nailfold_keypoint.utils.utils_visualize import \
    draw_kps_on_image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class KeypointTestDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, idx):
        img_original = self.imgs[idx]
        img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        target = {}
        target["image_id"] = torch.tensor([idx])

        return img, target

    def __len__(self):
        return len(self.imgs)


class KeypointDataset(Dataset):
    """
    关键点数据集
    """
    def __init__(self, root, keypoint_num, transform=None, demo=False, test=False):
        self.test = test
        self.root = root
        self.transform = transform
        self.keypoint_num = keypoint_num
        # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.demo = demo
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        # self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
        self.annotations_files = [x.replace('images', 'annotations').replace('.png', '.json').replace('.jpg', '.json') for x in self.imgs_files]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(
            self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        if not self.test:
            with open(annotations_path) as f:
                data = json.load(f)
                bboxes_original = data['bboxes']
                keypoints_original = data['keypoints']
                if 'classes' in data:
                    classes_original = data['classes']
                else:
                    # All objects are glue tubes
                    classes_original = [1 for _ in bboxes_original]
        else:
            bboxes_original = []
            keypoints_original = []

            # All objects are glue tubes
            classes_original = []
        bboxes = []
        while self.transform and len(bboxes)==0:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2]
                                            for kp in keypoints_original for el in kp]            # Apply augmentations
            sequence = np.arange(len(keypoints_original_flattened))
            bbox_sequence = np.arange(len(bboxes_original))
            transformed = self.transform(image=img_original, bboxes=bboxes_original,
                                         bboxes_labels=bbox_sequence, 
                                         keypoints=keypoints_original_flattened,
                                         kp_labels=sequence)
            img = transformed['image']
            bboxes_uncut = transformed['bboxes']

            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            # 6 keypoints now ,so 6,2 instead of 2,2
            keypoints_uncut = transformed['keypoints']
            bbox_sequence = np.array(transformed['bboxes_labels'])
            sequence = transformed['kp_labels']
            o_idx = -1
            # 一组关键点
            kps = []
            # 图片中所有关键点组
            keypoints = []
            bboxes = []
            for i,seq in enumerate(sequence):
                now_idx = int(seq / self.keypoint_num)
                kp_idx = int(seq % self.keypoint_num)
                if now_idx != o_idx:
                    if len(kps) == self.keypoint_num:
                        keypoints.append(kps)
                        bboxes.append(bboxes_uncut[np.where(bbox_sequence==o_idx)[0].item()])
                    # new one
                    o_idx = now_idx
                    kps = []
                kps.append(
                    list(keypoints_uncut[i]) + [keypoints_original[now_idx][kp_idx][2]])
            if len(kps) == self.keypoint_num:
                keypoints.append(kps)
                bboxes.append(bboxes_uncut[np.where(bbox_sequence==o_idx)[0].item()])
            # keypoints_transformed_unflattened = np.reshape(
            #     np.array(keypoints_uncut), (-1, self.keypoint_num, 2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            assert(len(bboxes)==len(keypoints))
            for i,bbox in enumerate(bboxes):
                assert(kp_in_bbox(keypoints[i][0], bbox))
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

        # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["image_id"] = torch.tensor([idx])
        img = F.to_tensor(img)
        if not self.test:
            target["boxes"] = bboxes
            target["labels"] = torch.as_tensor(
                classes_original, dtype=torch.int64) 
            
            target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * \
                (bboxes[:, 2] - bboxes[:, 0])
            target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
            target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
            

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["image_id"] = torch.tensor([idx])
        img_original = F.to_tensor(img_original)
        if not self.test:
            target_original["boxes"] = bboxes_original
            target_original["labels"] = torch.as_tensor(
                classes_original, dtype=torch.int64)  # all objects are glue tubes
            target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                bboxes_original[:, 2] - bboxes_original[:, 0])
            target_original["iscrowd"] = torch.zeros(
                len(bboxes_original), dtype=torch.int64)
            target_original["keypoints"] = torch.as_tensor(
                keypoints_original, dtype=torch.float32)
        
        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)

    def target2numpy(self, target):
        """
        Return: bboxes, keypoints, labels
        """
        bboxes = target[0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
        keypoints = []
        for kps in target[0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
            my_kps = []
            for kp in kps:
                if(kp[2]>0.5):
                    my_kps.append(kp[:2])
            keypoints.append(my_kps)
        labels = target[0]['labels'].detach().cpu().numpy()
        return bboxes, keypoints, labels
    
    def visualize(self, name, image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, scores=None, output_dir=None, labels=None):
        # if output_dir is None:
        #     output_dir = f"./log/{self.expname}/figs"
        # os.makedirs(output_dir,exist_ok=True)
        fontsize = 5

        for id,bbox in enumerate(bboxes):
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 1)
            if scores is not None:
                image = cv2.putText(image.copy(), " " + str(round(float(scores[id]),2)), start_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            if labels is not None:
                image = cv2.putText(image.copy(), " " + str(labels[id]), start_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp), 1, (255,255,255), 1)
                # image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        if image_original is None and keypoints_original is None:
            plt.figure(figsize=(20, 10))
            plt.imshow(image)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir,name+'.png'))
            plt.close()

        else:
            for id,bbox in enumerate(bboxes_original):
                start_point = (bbox[0], bbox[1])
                end_point = (bbox[2], bbox[3])
                color = [255,0,0]
                if labels is not None:
                    color = [0,0,0]
                    color[labels[id]%3] = 255
                image_original = cv2.rectangle( image_original.copy(), start_point, end_point, color, 1)
                if labels is not None:
                    image_original = cv2.putText(image_original.copy(), " " + str(labels[id]), start_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            for kps in keypoints_original:
                for idx, kp in enumerate(kps):
                    image_original = cv2.circle(image_original.copy(), tuple(kp), 1, (255,255,255), 1)
                    # image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(
                    #     kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

            f, ax = plt.subplots(1, 2, figsize=(20, 10))

            ax[0].imshow(image_original)
            ax[0].set_title('Original image', fontsize=fontsize)

            ax[1].imshow(image)
            ax[1].set_title('Transformed image', fontsize=fontsize)
            plt.savefig(f"./Keypoint_Detection/nailfold_keypoint/model/{name}.png")
            print(f"save to {f}")
            # plt.savefig(f"./log/{self.expname}/figs/"+name+'.png')
            plt.close()


def kp_in_bbox(kp, bbox):
    """
    kp: (x,y)
    bbox: [xmin, ymin, xmax, ymax]
    """
    if kp[0] < bbox[0] or kp[0] > bbox[2]:
        return False
    if kp[1] < bbox[1] or kp[1] > bbox[3]:
        return False
    return True


def kp_train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomCrop(always_apply=False, p=0.7, height=random.randrange(400,600), width=random.randrange(600,800)),
            # Random rotation of an image by 90 degrees zero or more times
            A.RandomRotate90(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True,
                                       always_apply=False, p=0.3),  # Random change of brightness & contrast
        ], p=1)
    ],
        # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        keypoint_params=A.KeypointParams(format='xy', label_fields=['kp_labels']),
        # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['bboxes_labels'])
    )


class MaskDataset_RCNN(Dataset):
    """
    实例分割数据集
    """
    def __init__(self, root, class_name_dict, transform=None, demo=False, test=False):
        self.test = test
        self.root = root
        self.transform = transform
        self.class_name_dict = class_name_dict
        self.PADDING = 10
        # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.demo = demo
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
        if not self.test:
            with open(annotations_path) as f:
                data = json.load(f)
                # 所有多边形
                plys = [shape['points'] for shape in data['shapes']]
                classes_original = [self.class_name_dict[shape['label']] for shape in data['shapes']]
                zeros = np.zeros((h,w), dtype=np.uint8)
                masks_original = []
                bboxes_original = []
                for ply in plys:
                    # 多边形转mask
                    ply = np.array(ply).astype(np.int32)
                    mask = cv2.fillPoly(zeros.copy(), [ply], 1)
                    masks_original.append(mask)
                    # 计算包围盒
                    xmin = max(0, ply.min(axis=0)[0] - self.PADDING)
                    xmax = min(w, ply.max(axis=0)[0] + self.PADDING)
                    ymin = max(0, ply.min(axis=0)[1] - self.PADDING)
                    ymax = min(h, ply.max(axis=0)[1] + self.PADDING)
                    bbox = [xmin, ymin, xmax, ymax]
                    bboxes_original.append(bbox)
        else:
            masks_original = []
            bboxes_original = []

            # All objects are glue tubes
            classes_original = []
        if self.transform:
            # Apply augmentations
            sequence = np.arange(len(bboxes_original))
            transformed = self.transform(image=img_original, bboxes=bboxes_original,
                                         bboxes_labels=classes_original, masks=masks_original, masks_labels=sequence)
            img = transformed['image']
            bboxes = transformed['bboxes']
            old_masks = transformed['masks']
            sequence = transformed['masks_labels']
            masks = []
            assert(len(old_masks)==len(masks_original))
            for i in sequence:
                masks.append(old_masks[i])
            assert(len(masks)==len(bboxes))
            labels = transformed['bboxes_labels']
            if len(labels)==0:
                img, bboxes, masks, labels = img_original, bboxes_original, masks_original, classes_original

        else:
            img, bboxes, masks, labels = img_original, bboxes_original, masks_original, classes_original
        bboxes = np.array(bboxes)
        masks = np.array(masks)
        # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["image_id"] = torch.tensor([idx])
        img = F.to_tensor(img)
        if not self.test:
            target["boxes"] = bboxes
            target["labels"] = torch.as_tensor(
                labels, dtype=torch.int64) 
            
            target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * \
                (bboxes[:, 2] - bboxes[:, 0])
            target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
            target["masks"] = torch.as_tensor(masks, dtype=torch.bool)
            
        bboxes_original = np.array(bboxes_original)
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["image_id"] = torch.tensor([idx])
        img_original = F.to_tensor(img_original)
        masks_original = np.array(masks_original)
        if not self.test:
            target_original["boxes"] = bboxes_original
            target_original["labels"] = torch.as_tensor(
                classes_original, dtype=torch.int64)  # all objects are glue tubes
            target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                bboxes_original[:, 2] - bboxes_original[:, 0])
            target_original["iscrowd"] = torch.zeros(
                len(bboxes_original), dtype=torch.int64)
            target_original["masks"] = torch.as_tensor(
                masks_original, dtype=torch.bool)
        
        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)

    def target2numpy(self, target):
        """
        将一个batch的数据转为可视化用的numpy数组
        Return: bboxes, masks, labels
        """
        bboxes = target[0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
        masks = target[0]['masks'].detach().cpu().numpy()
        labels = target[0]['labels'].detach().cpu().numpy()
        return bboxes, masks, labels


def mask_train_transform():
    return A.Compose([
        A.Sequential([
            # Random rotation of an image by 90 degrees zero or more times
            A.RandomCrop(always_apply=False, p=0.7, height=random.randrange(400,600), width=random.randrange(600,800)),
            A.RandomRotate90(p=1),
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True,
            #                            always_apply=False, p=1),  # Random change of brightness & contrast
        ], p=1)
    ],
        # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['bboxes_labels', 'masks_labels'])
    )

