import argparse
import typing

import cv2
import numpy as np
import torch
import yaml
from Image_Segmentation.image_segmentation.solver import Solver
from Image_Segmentation.image_segmentation.utils.image_splitter import \
    ImageSplitter
from torchvision import transforms as T


def t_images2masks(images:typing.List[np.ndarray], split_num: int = 2, pad_ratio: float= 1)->np.ndarray:
    '''
    segment the images, split it into patches(split_num) and then segment to get higher resolution
    args
        images: list[np.ndarray], [(height, width, chanels)]
        split_num: int, in order to get higher resolution, split the image into split_num*split_num patches
    return
        segments: np.ndarray, (batch_size, height, width), same size as the input image, 0, 1 only
    '''
    # Load the config file
    with open("./Image_Segmentation/image_segmentation/config.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)
    
    # Build image segmentation solver
    solver = Solver(config)

    h,w,c = images[0].shape

    # Split the image into patches
    splitter = ImageSplitter(images[0].shape[0], images[0].shape[1], config.img_ch, num_patches=split_num)
    encoder = splitter.split_into_patch
    decoder = splitter.reverse

    images_ls = []
    for image in images:
        if config.img_ch==1:    
            images_ls += encoder(image[:,:,1:2])
        elif config.img_ch==3:
            images_ls += encoder(image)
    images = images_ls

    h1, w1, c1 = images[0].shape
    pad_h = int(h1*(pad_ratio-1)/2)
    pad_w = int(w1*(pad_ratio-1)/2)

    # Prepocessing the images
    Transform = []
    Transform.append(T.ToTensor())
    if config.img_ch==1:
        Transform.append(T.Normalize((0.5), (0.5)))
    else:
        Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    Transform.append(T.Pad((pad_h, pad_w, pad_h, pad_w)))
    Transform.append(T.Resize((int(config.image_size/w*h),config.image_size)))
    Transform = T.Compose(Transform)

    images = [Transform(img) for img in images]
    images = torch.stack(images)

    # Segmentation and joint into the full image
    masks = solver.images2masks(images)

    # unpad
    h3, w3 = masks.shape[2], masks.shape[3]
    h2 = h1+pad_h*2
    w2 = w1+pad_w*2
    pad_h2 = int(pad_h/h2*h3)
    pad_w2 = int(pad_w/w2*w3)
    masks = np.array([masks[i,:,pad_h2:-pad_h2-1,pad_w2:-pad_w2-1] for i in range(len(masks))])
    masks = decoder(masks)
    masks = masks[:,0,:,:] # delete the channel dimension

    # resize to the original size
    segments = np.zeros([len(masks), h, w], dtype=np.uint8)
    for i in range(len(masks)):
        segments[i,:,:] = cv2.resize(masks[i], (w,h)).astype(np.uint8)

    return segments


