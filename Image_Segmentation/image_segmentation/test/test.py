"""
测试模型性能
"""
import argparse
import os

import cv2
import numpy as np
from Image_Segmentation.image_segmentation.evaluation import *
from Image_Segmentation.image_segmentation.image2segment import t_images2masks
from Image_Segmentation.image_segmentation.solver import Solver
from Image_Segmentation.image_segmentation.test.test_dataset import (
    DataLoader, SegmentDataset)
from torchvision import transforms as T


def test(unet_path, scale, visualize=True):
    img_ch = 3
    config = argparse.Namespace(augmentation_prob=0.4, batch_size=1, beta1=0.5, 
                                beta2=0.999, cuda_idx=1, image_size=224, img_ch=img_ch, 
                                log_step=2, lr=0.0002, mode='train', model_path='./checkpoints', 
                                model_type='U_Net', num_epochs=100, num_epochs_decay=70, 
                                num_workers=8, output_ch=1, result_path='./result/', 
                                t=3, test_path='../data/For-Segmentation/', 
                                train_path='../data/For-Segmentation/', val_step=2, 
                                valid_path='../data/For-Segmentation/')
    
    output_dir = os.path.join("test/log", f"{unet_path[11:20]}-{scale}")
    data_dir = "./Data_Preprocess/nailfold_data_preprocess/data/instance_segment_dataset"
    dataset = SegmentDataset(os.path.join(data_dir,"test"), ['normal', 'abnormal']) #, 'blur'
    data_loader = DataLoader(dataset, batch_size=8,
                                shuffle=True)
    solver = Solver(config, None, None, None)
    os.makedirs(output_dir, exist_ok=True)
    all_sensitivity = []
    all_precision = []
    for img_path, images, gt in data_loader:
        origin_imgs = [cv2.imread(img_p) for img_p in img_path]
        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        images = Norm_(images)
        # 取G通道
        if img_ch==1:
            images = images[:,1:2,...]
        masks = solver.t_images2masks(images, unet_path=unet_path, scale=scale)
         
        for mask,gt_,origin_img,img_p in zip(masks, gt, origin_imgs, img_path):
            
            result = test_acc(torch.Tensor(mask), gt_)
            
            if gt_.any():
                print(result)
                all_precision.append(result['precision'])
                all_sensitivity.append(result['sensitivity'])
            if visualize:
                mask = mask > 0.5
                mask_vis = mask.astype(np.uint8) * 255
                gt_vis = gt_.numpy().astype(np.uint8) * 255
                vis = np.concatenate([origin_img, cv2.cvtColor(mask_vis,cv2.COLOR_GRAY2BGR), cv2.cvtColor(gt_vis,cv2.COLOR_GRAY2BGR)])
                cv2.putText(vis, f" pr:{round(result['precision'], 2)} se:{round(result['sensitivity'], 2)}", (100,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(output_dir, os.path.basename(img_p)) ,vis)
    print(f"===SE:{np.array(all_sensitivity).mean()}===PR:{np.array(all_precision).mean()}===")

def test_acc(masks, gt):
    sensitivity = get_sensitivity(masks, gt)
    precision = get_precision(masks, gt)
    return {
        "sensitivity": sensitivity,
        "precision": precision
    }

if __name__ == "__main__":
    # test(unet_path="test_model/U_Net-60-0.0002-50-0.7000.pkl", scale=2.8, visualize=True)
    test(unet_path="test_model/U_Net-20-0.0002-50-0.7000.pkl", scale=3, visualize=True)