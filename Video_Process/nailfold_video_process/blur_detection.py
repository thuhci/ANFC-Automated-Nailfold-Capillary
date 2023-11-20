
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Video_Process.nailfold_video_process.utils.process_video import *

PADDING = 30
kp_threshold = 7.5
img_threshold = 5

def estimate_blur(image: np.array, mask, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    # blur_map[np.abs(blur_map) < 5] = 0
    score = np.var(blur_map[mask])
    blur_map[~mask] = 0
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map: np.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = np.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


def visualize_blur_map(blur_map):
    min = -80
    max = 160 # blur_map.max()
    blur_map = blur_map - min
    # blur_map = blur_map * 255 / max
    return blur_map.astype(np.uint8)


def detect_kps():
    dir = "./Video_Process/data/aligned_videos/7.28/55896"
    name = "wmv1"
    file = os.path.join(dir, f"{name}.mp4")
    imgs,fps = process_video(file,1)
    with open(os.path.join(dir,f"{name}-kp.json"),"r") as f:
        kp = json.load(f)
    kp = kp['kp'][0]
    bboxs = [np.concatenate([np.array(p[0]) - np.array([PADDING,10]), np.array(p[0]) +  np.array([PADDING,2*PADDING])]) for p in kp]
    blur_scores = np.zeros((len(imgs),len(bboxs)))
    newimgs = []

    # segment_input = os.path.join("./debug", f'{name}-segments.mp4')
    # if os.path.exists(segment_input):
    #     segment_video, _ = process_video(segment_input, 1)
    #     segments = [cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) for seg in segment_video]
    # else:
    #     segments = video2segments(imgs)
    #     segment_video = [cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) for seg in segments]
    #     imgs2video(segment_video, segment_input)

    for i,img in enumerate(imgs):
        newimg = img.copy()
        for j,bbox in enumerate(bboxs):
            block = img[max(round(bbox[1]),0):round(bbox[3]),max(round(bbox[0]),0):round(bbox[2])]
            # mask = segments[i][max(round(bbox[1]),0):round(bbox[3]),max(round(bbox[0]),0):round(bbox[2])]
            block = block[...,1]
            # mask = cv2.dilate(mask, np.ones((10,10),np.uint8))
            mask = np.ones_like(block)
            block = cv2.morphologyEx(block, cv2.MORPH_OPEN ,np.ones((7,7),np.uint8))
            blur_map, score, blurry = estimate_blur(block, mask.astype(np.bool8))
            h,w = blur_map.shape
            newimg[bbox[3]:bbox[3]+h, bbox[0]:bbox[0]+w] = cv2.cvtColor(visualize_blur_map(blur_map),cv2.COLOR_GRAY2BGR)
            # newimg[bbox[3]+h:bbox[3]+2*h, bbox[0]:bbox[0]+w] = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            if score < kp_threshold:
                cv2.rectangle(newimg, (bbox[0],bbox[1]), (bbox[2],bbox[3]),(255,0,0),1)
            else:
                cv2.rectangle(newimg, (bbox[0],bbox[1]), (bbox[2],bbox[3]),(0,255,0),1)
            cv2.putText(newimg, f"{round(score,2)}", (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
            blur_scores[i][j] = score
        newimgs.append(newimg)
        
    # blur_scores = np.array(blur_scores)
    color= ['r','g','b','purple']
    print(blur_scores.shape)
    for i in range(blur_scores.shape[1]):
        plt.plot(np.arange(blur_scores.shape[0]), blur_scores[:,i], c=color[i%4])
    plt.savefig("b.png")
    imgs2video(newimgs,"a.mp4")

import json

if __name__ == '__main__':
    dir = "./Video_Process/data/aligned_videos/7.28/58497"
    name = "wmv2-kp"
    file = os.path.join(dir, f"{name}.mp4")
    imgs,fps = process_video(file,1)

    newimgs = []
    blur_scores = np.zeros(len(imgs))
    mask = np.ones_like(imgs[0][...,0])
    for i,img in enumerate(imgs):
        newimg = img.copy()
        block = img[...,1]
        block = cv2.morphologyEx(block, cv2.MORPH_OPEN ,np.ones((7,7),np.uint8))

        blur_map, score, blurry = estimate_blur(block, mask.astype(np.bool8))
        h,w = blur_map.shape
        blur_scores[i] = score

        # newimg[bbox[3]+h:bbox[3]+2*h, bbox[0]:bbox[0]+w] = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        if score < img_threshold:
            cv2.rectangle(newimg, (1,1), (w-1,h-1),(255,0,0),1)
        else:
            cv2.rectangle(newimg, (1,1), (w-1,h-1),(0,255,0),1)
        cv2.putText(newimg, f"{round(score,2)}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        newimgs.append(newimg)
    
    # blur_scores = np.array(blur_scores)
    color= ['r','g','b','purple']
    print(blur_scores.shape)
    plt.plot(np.arange(blur_scores.shape[0]), blur_scores[:], c=color[i%4])
    plt.savefig("b.png")
    imgs2video(newimgs,"a.mp4")