import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm
from Keypoint_Detection.nailfold_keypoint.dataset import (KeypointDataset,
                                                          kp_train_transform)
from Keypoint_Detection.nailfold_keypoint.engine import (_get_state_dict,
                                                         evaluate,
                                                         train_one_epoch)
from Keypoint_Detection.nailfold_keypoint.utils.utils import collate_fn
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F


def get_kp_rcnn_model(model_name):
    '''
    This function returns keypointrcnn_resnet50_fpn
    input:
        num_keypoints: number of keypoints
        num_classes: number of classes
        weights_path: path to weights file
    '''
    state_dict, keypoint_num, class_num = _get_state_dict(model_name)

    anchor_generator = AnchorGenerator(sizes=(
        32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                pretrained_backbone=True,
                                                                num_keypoints=keypoint_num,
                                                                num_classes=class_num,  # Background is the first class, object is the second class
                                                                rpn_anchor_generator=anchor_generator)

    model.load_state_dict(state_dict)
    return model


class Keypoint():
    def __init__(self, cfg_dict) -> None:
        self.dataset_dir = cfg_dict.get("dataset_dir")
        self.class_name_dict = cfg_dict.get("class_name_dict")
        self.expname = cfg_dict.get("expname")

        self.num_epochs = cfg_dict.get("num_epochs")
        self.bbox_th = cfg_dict.get("bbox_th")

        keypoints_classes_ids2names = cfg_dict.get("keypoints_classes_ids2names")
        self.keypoint_num = len(keypoints_classes_ids2names)

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

        if image_original is None or keypoints_original is None:
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
            plt.savefig(f"./log/{name}.png")
            print(f"save to {f}")
            # plt.savefig(f"./log/{self.expname}/figs/"+name+'.png')
            plt.close()

    def train(self, writer=None):
        trainset_path = os.path.join(self.dataset_dir, 'train')
        testset_path = os.path.join(self.dataset_dir, 'test')

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        trainset = KeypointDataset(trainset_path, keypoint_num=self.keypoint_num, transform=kp_train_transform(), demo=False)
        testset = KeypointDataset(testset_path, keypoint_num=self.keypoint_num, transform=None, demo=False)

        train_loader = DataLoader(
            trainset, batch_size=3, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(
            testset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        model = get_kp_rcnn_model(num_keypoints=self.keypoint_num, num_classes=len(self.class_name_dict))
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5)
        num_epochs = self.num_epochs

        os.makedirs(f"./log/{self.expname}/ckpts",exist_ok=True)

        for epoch in range(num_epochs):

            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10,writer=writer)
            lr_scheduler.step()

            evaluate(model, test_loader, device, keypoint_num=self.keypoint_num)
            
            if (epoch+1) % 10 == 0 or epoch == 49:
                state_dict = model.state_dict()
                state_dict['keypoint_num'] = self.keypoint_num
                state_dict['class_num'] = len(self.class_name_dict)
                torch.save(state_dict, f'./log/{self.expname}/ckpts/rcnn_weights_epo_{epoch}.pth')


    def test(self, visualize=True):
        """测试

        Args:
            visualize (bool, optional): _description_. Defaults to True.
        Returns:
            {
            "img_path": all_path,
            "output": all_keypoints,
            "gt": all_keypoints_gt
            }
        """
        testset_path = os.path.join(self.dataset_dir, 'test')
        dataset_test = KeypointDataset(testset_path, keypoint_num=self.keypoint_num, transform=None, demo=False)
        data_loader_test = DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = get_kp_rcnn_model(num_keypoints=self.keypoint_num, num_classes=len(self.class_name_dict))
        model.to(device)

        epoch = 100
        while epoch>0:
            model_path = f'./log/{self.expname}/ckpts/rcnn_weights_epo_{epoch}.pth'
            if os.path.exists(model_path):
                break
            epoch-=1
        state_dict = torch.load(model_path)
        if "keypoint_num" in state_dict:
            now_keypoint_num = state_dict["keypoint_num"]
            del state_dict["keypoint_num"]

        if "class_num" in state_dict:
            now_class_num = state_dict["class_num"]
            del state_dict["class_num"]
            model = get_kp_rcnn_model(num_keypoints=now_keypoint_num, num_classes=now_class_num)
            model.to(device)
        model.load_state_dict(state_dict)
        iterator = tqdm.tqdm(iter(data_loader_test))
        cnt = 0

        all_keypoints = []
        all_keypoints_gt = []
        all_path = []
        all_bboxes = []

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

            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))

            labels = []
            for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                labels.append(label)

            all_bboxes.append(bboxes)

            all_keypoints.append(keypoints)
            all_keypoints_gt.append(targets[0]['keypoints'])
            all_path.append(dataset_test.imgs_files[targets[0]['image_id']])
            if visualize:
                self.visualize(f"test_{cnt}", image, bboxes, keypoints, scores=scores, labels=labels)
        
        return {
            "img_path": all_path,
            "output": all_keypoints,
            "bbox": all_bboxes,
            "gt": all_keypoints_gt
        }




