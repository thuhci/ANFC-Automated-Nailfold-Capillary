import json
import os

import albumentations as A  # Library for augmentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm
from Keypoint_Detection.nailfold_keypoint.dataset import (MaskDataset_RCNN,
                                                          mask_train_transform)
from Keypoint_Detection.nailfold_keypoint.engine import (_get_state_dict,
                                                         evaluate,
                                                         train_one_epoch)
from Keypoint_Detection.nailfold_keypoint.utils.utils import collate_fn
from Keypoint_Detection.nailfold_keypoint.utils.utils_visualize import \
    draw_mask_on_image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F


def get_mask_rcnn_model(model_name):
    state_dict, keypoint_num, class_num = _get_state_dict(model_name)

    anchor_generator = AnchorGenerator(sizes=(
        32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                                pretrained_backbone=True,
                                                                num_classes=class_num,  # Background is the first class, object is the second class
                                                                rpn_anchor_generator=anchor_generator)
    model.load_state_dict(state_dict)
    return model


class Mask():
    def __init__(self, cfg_dict) -> None:
        self.dataset_dir = cfg_dict.get("dataset_dir")
        self.class_name_dict = cfg_dict.get("class_name_dict")
        self.expname = cfg_dict.get("expname")

        self.num_epochs = cfg_dict.get("num_epochs")
        self.bbox_th = cfg_dict.get("bbox_th")

        self.mask_colors = cfg_dict.get("mask_colors")

    def images_compare(self, name, image_original, image):
        """
        对比两张图片
        """
        fontsize = 10
        f, ax = plt.subplots(1, 2, figsize=(20, 10))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
        plt.savefig(f"./log/{self.expname}/figs/"+name+'.png')
        plt.close()

    def visualize(self, name, image, bboxes, masks, image_original=None, bboxes_original=None, masks_original=None, scores=None, output_dir=None, labels=None, labels_original=None):
        if output_dir is None:
            output_dir=f"./log/{self.expname}/figs"
        fontsize = 5
        os.makedirs(output_dir,exist_ok=True)

        image = draw_mask_on_image(image, bboxes, masks, scores, labels)  
                
        if image_original is None and masks_original is None:
            plt.figure(figsize=(20, 10))
            plt.imshow(image)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir,name+'.png'))
            plt.close()

        else:
            image_original = draw_mask_on_image(image_original, bboxes_original, masks_original, scores, labels_original)
            f, ax = plt.subplots(1, 2)

            ax[0].imshow(image_original)
            ax[0].set_title('Original image', fontsize=fontsize)

            ax[1].imshow(image)
            ax[1].set_title('Transformed image', fontsize=fontsize)
            plt.savefig(f"./log/{self.expname}/figs/"+name+'.png')
            plt.close()

    def train(self, writer=None):
        data_path = self.dataset_dir # "./Data_Preprocess/data/segment_dataset_clean"
        KEYPOINTS_FOLDER_TRAIN = os.path.join(data_path, 'train')

        dataset = MaskDataset_RCNN(KEYPOINTS_FOLDER_TRAIN,
                            transform=mask_train_transform(), demo=True)
        data_loader = DataLoader(dataset, batch_size=1,
                                shuffle=True, collate_fn=collate_fn)

        iterator = iter(data_loader)
        # print("Original targets:\n", batch[3], "\n\n")
        # print("Transformed targets:\n", batch[1])

        for id,batch in enumerate(iterator):
            image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

            masks = batch[1][0]['masks'].detach().cpu().numpy()

            image_original = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bboxes_original, masks_original, labels_original = dataset.target2numpy(batch[3])

            labels = batch[1][0]['labels'].detach().cpu().numpy().astype(np.int32).tolist()

            self.visualize(f"show_{id}", image, bboxes, masks, image_original,
                    bboxes_original, masks_original, labels=labels, labels_original=labels_original)
            if id > 3:
                break

        ############################## Train ################################
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        KEYPOINTS_FOLDER_TRAIN = os.path.join(data_path, 'train')
        KEYPOINTS_FOLDER_TEST = os.path.join(data_path, 'test')

        dataset_train = MaskDataset_RCNN(
            KEYPOINTS_FOLDER_TRAIN, transform=mask_train_transform(), demo=False)
        dataset_test = MaskDataset_RCNN(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

        data_loader_train = DataLoader(
            dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)
        data_loader_test = DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        model = get_mask_rcnn_model(len(self.class_name_dict))
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5)
        num_epochs = self.num_epochs

        os.makedirs(f"./log/{self.expname}/ckpts",exist_ok=True)
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10,writer=writer)
            lr_scheduler.step()
            evaluate(model, data_loader_test, device, keypoint_num=1)
            
            if (epoch+1) % 25 == 0 or epoch == 49:
                state_dict = model.state_dict()
                state_dict['class_num'] = len(self.class_name_dict)
                torch.save(state_dict, f'./log/{self.expname}/ckpts/rcnn_weights_epo_{epoch}.pth')

        ############################# Train Complete ##################################
        epoch = 0
        while epoch<100:
            model_path = f'./log/{self.expname}/ckpts/rcnn_weights_epo_{epoch}.pth'
            if os.path.exists(model_path):
                break
            epoch+=1
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        iterator = iter(data_loader_test)
        iterator = tqdm.tqdm(iterator)
        cnt = 0
        for images, targets in iterator:
            cnt+=1
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                model.to(device)
                model.eval()
                output = model(images)

            # print("Predictions: \n", output)


            image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = output[0]['scores'].detach().cpu().numpy()

            # Indexes of boxes with scores > 0.7
            high_scores_idxs = np.where(scores > self.bbox_th)[0].tolist()
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

            # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
            # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
            # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

            masks = []
            for mask in output[0]['masks'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                masks.append((mask[0]>0.5).astype(np.bool8))

            bboxes = []
            for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))

            labels = output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
            scores = output[0]['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()

            bboxes_target, masks_target, labels_target = dataset_test.target2numpy(targets)

            output = draw_mask_on_image(image.copy(), bboxes, masks, scores, labels)
            target = draw_mask_on_image(image.copy(), bboxes_target, masks_target, None, labels_target)
            self.images_compare(f"test_{cnt}", target, output)
