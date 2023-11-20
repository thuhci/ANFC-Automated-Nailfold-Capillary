import argparse
import os
import sys

import cv2
import numpy as np
import tqdm


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input', default=None, type=str)
    parser.add_argument('--output', help='video to store', dest='output', default="./Video_Process/data/aligned_videos", type=str)
    #default为间隔多少帧截取一张图片
    parser.add_argument('--skip_frame', dest='skip_frame', help='skip number of video', default=1, type=int)
    parser.add_argument('--type', dest='type', help='type of utils', default='', type=str)
    #input为输入视频的路径 ，output为输出存放图片的路径
    args = parser.parse_args(sys.argv[1:])
        # ['--input','./49510','--output','./Video_Process/data/aligned_videos']
    return args

def process_video(i_video, num):
    """将视频输出为图像列表

    Args:
        i_video (_type_): _description_
        num (_type_): 跳过帧数

    Returns:
        _type_: _description_
    """
    if not os.path.exists(i_video):
        return [],0
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) >= 3:
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # print("frame",num_frame)
    # print("fps",fps)
    expand_name = '.mp4'
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    count = 0
    imgs = []
    it = tqdm.tqdm(range(int(num_frame) + 1))
    it.set_description(f"video->imgs,{os.path.basename(i_video)}")
    for i in it:
        ret, frame = cap.read()
        if not ret:
            break
        if cnt % num == 0:
            count += 1
            imgs.append(frame)
        cnt += 1

    cap.release()
    return imgs,fps

def imgs2video(imgs, output_filename, fps=20):
    size = imgs[0].shape[:2]
    size = (size[1],size[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(output_filename, fourcc, fps, size)
    it = tqdm.tqdm(imgs)
    it.set_description(f"imgs->video,{os.path.basename(output_filename)}")
    for img in it:
        videowrite.write(img.astype(np.uint8))
    videowrite.release()