import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

mask_colors = [
        [255,255,255],
        [255,255,255],
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [255,255,0],
        [255,0,255],
    ]


def draw_kps_on_image(input, keypoints, color=(255,255,255), bboxes=None, scores=None, labels=None):
    """
    将识别出的bbox、keypoint绘制在图片上, 返回图片的numpy数组
    Return: new_image
    """
    image = input #(input.astype(np.int32).copy()*3/4).astype(np.uint8)
    for kps in keypoints:
        try:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp), 2, color, 2)
        except:
            image = cv2.circle(image.copy(), tuple(kps), 2, color, 2)
        # sp = (kps[0][0]-40, kps[0][1]-50)
        # ep = (kps[0][0]+40, kps[0][1]+150)
        # image = cv2.rectangle( image.copy(), sp, ep, [0,155,155], 1)
    # for id,bbox in enumerate(bboxes):
    #     start_point = (bbox[0], bbox[1])
    #     end_point = (bbox[2], bbox[3])
    #     if labels is not None:
    #         color = [0,0,0]
    #         color[labels[id]%3] = 255
    #         image = cv2.rectangle( image.copy(), start_point, end_point, color, 1)
    #     if scores is not None:
    #         image = cv2.putText(image.copy(), " " + str(round(float(scores[id]),2)), start_point, 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    #     if labels is not None:
    #         image = cv2.putText(image.copy(), " " + str(labels[id]), end_point, 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return image

class_color = {
    # abnormal is red, cross is blue, blur is grey, unnecessary is light grey, hemo is purple
    'normal': [0,102,0],
    'blur': [178,178,178],
    'abnormal': [0,51,204],
    'cross': [204,102,51],
    'unnecessary': [122,133,174],
    'hemo': [153,0,153]
}

def draw_mask_on_image(input, bboxes, masks, scores=None, labels=None, extra_info=None, gt_labels=None):
    """
    label: list(int/str)
    extra_info: list(str)
    Return: new_image
    """
    image = (input.astype(np.int32).copy()*3/4).astype(np.uint8)
    for id,bbox in enumerate(bboxes):
        start_point = (min(image.shape[1],bbox[0]), min(image.shape[0],bbox[1]))
        end_point =   (min(image.shape[1],bbox[2]), min(image.shape[0],bbox[3]))
        if labels is not None:
            if isinstance(labels[id],(int)) or isinstance(labels[id],(np.int64)):
                color = mask_colors[labels[id]]
            else:
                color = class_color[labels[id]]
            thickness = 2 if labels[id] in ['normal','abnormal','cross'] else 1
            image = cv2.rectangle(image.copy(), start_point, end_point, color, thickness)

            image = cv2.putText(image.copy(), " " + str(labels[id]), start_point, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            if masks is not None:
                image[masks[id]] += (np.array(mask_colors[labels[id]]) / 6).astype(np.uint8)
        if extra_info is not None:
            image = cv2.putText(image.copy(), " " + str(extra_info[id]), (bbox[0], bbox[3]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        if gt_labels is not None:
            image = cv2.putText(image.copy(), " " + str(gt_labels[id]), (bbox[2], bbox[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return image

def draw_line(image, line, color):
    image = cv2.line(image.copy(), line[0], line[1], color, 1)
    # two short vertical line at the end of the line
    image = cv2.line(image.copy(), line[0], (line[0][0], line[0][1]+5), color, 3)
    image = cv2.line(image.copy(), line[0], (line[0][0], line[0][1]-5), color, 3)
    image = cv2.line(image.copy(), line[1], (line[1][0], line[1][1]+5), color, 3)
    image = cv2.line(image.copy(), line[1], (line[1][0], line[1][1]-5), color, 3)

    return image