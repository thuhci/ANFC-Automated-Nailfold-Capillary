#!/usr/bin/python

"""
Phase Correlation implementation in Python
Michael Ting
Created 25 June 2013
Updated 8 July 2013

Algorithm:
    Given two input images A and B:
    Apply window function on both A and B to reduce edge effects
    Calculate the discrete 2D Fourier transform of both A and B
        G_a = F{A}
        G_b = F{B}
    Calculate the cross-power spectrum by taking the complex conjugate of G_b,
        multiplying the Fourier transforms together elementwise, and normalizing
        the product elementwise
        R = (G_a %*% G_B*) / (|G_a G_b*|)
            %*% is the Hadamard (entry-wise) product
    Obtain normalized cross-correlation by applying the inverse Fourier transform
        r = F^-1{R}
    Determine the location of the peak in r:
        (del_x, del_y) = argmax over (x,y) of {r}
"""

import numpy as np
from scipy import misc
import cv2
from Video_Process.nailfold_video_process.utils.process_video import *
import json
import matplotlib.pyplot as plt
from Video_Process.nailfold_video_process.utils.align import *
from Video_Process.nailfold_video_process.tools import NailfoldTools

# a and b are numpy arrays
def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

def phase_affine(source_g, target_g, source=None):
    """相位相关法进行图像配准

    Args:
        source_g (np.array): 灰度图像
        target_g (np.array): 灰度图像
        source (np.array, optional): 彩色图像. Defaults to None.
        source_seg (np.array, optional): 彩色图像. Defaults to None.

    Returns:
        np.array[[],[]]: trans平移
        np.array: warp
        np.array: warp_seg
    """
    # Obtain the size of target image
    rows, cols, _ = source.shape
    
    # cv2.imwrite("b.png",target_g)
    result = phase_correlation(source_g, target_g)
    # result = np.max(result,axis=-1)
    trans = np.where(result==result.max())
    if trans[0][0] > rows / 2:
        trans[0][0] = trans[0][0] - rows
    if trans[1][0] > cols / 2:
        trans[1][0] = trans[1][0] - cols
    M = np.eye(2)
    # 需要调换长宽
    trans = [[trans[1][0]], [trans[0][0]]]
    trans = np.array(trans)
    trans = -trans
    # print(trans)
    M = np.concatenate([M,trans],axis=-1)
    # Warp the source image
    if source is not None:
        warp = cv2.warpAffine(source.copy(), M, (cols, rows))
    return trans, result, warp


def visualize_phase(R):
    R = R - R.min() + 10
    R = R * 255 / R.max()
    R[R < 200] = 0
    R = R.astype(np.uint8)
    R = cv2.cvtColor(R,cv2.COLOR_GRAY2BGR)
    return R

MOVE_VALID_PADDING = 20
def check_result0(R):
    R = R - R.min() + 10
    R = R * 255 / R.max()
    R[R < 200] = 0
    possible_move = np.where(R > 0)
    possible_move = np.array(possible_move)
    h,w = R.shape
    center_h = (possible_move[1] > MOVE_VALID_PADDING) * (possible_move[1] < h - MOVE_VALID_PADDING)
    center_w = (possible_move[0] > MOVE_VALID_PADDING) * (possible_move[0] < w - MOVE_VALID_PADDING)
    if np.count_nonzero(center_h) + np.count_nonzero(center_w) > 5:
        return True
    pass


def check_result(origin, new):
    origin = origin.astype(np.bool8).astype(int)
    new = new.astype(np.bool8).astype(int)
    mse = np.mean((origin - new) ** 2)
    return mse
    

def kpvideo2kpvideo_by_phase(args, video_i=None, start=0, end_cut = -1, threshold=0.2, use_batch=False, debug=False):
    """将视频做去抖

    Args:
        args (_type_): _description_
        video_i (int, optional): 若是None则全部做处理，否则指定一个视频做处理. Defaults to None.
        start (int, optional): _description_. Defaults to 0.
        end (int, optional): _description_. Defaults to -1.
        threshold (float, optional): _description_. Defaults to 0.2.
        debug (bool, optional): _description_. Defaults to False.
    """
    tool = NailfoldTools()
    with open(os.path.join(args.input,"kps.json"), "r") as file:
        json_kps = json.load(file)
    best_segment_i = video_i
    # ==========================Phase-Correlate======================
    if video_i is None:
        samples = []
        for i in range(len(json_kps['kp-videos'])):
            json_kp = json_kps['kp-videos'][i]
            filename = json_kp['path']
            imgs,fps = process_video(filename, 1)
            samples.append(imgs[0])
        if len(samples) == 0:
            return "no kp"
        segments = tool.kpimgs2segments(samples)
        segment_area = np.array([np.count_nonzero(segment) for segment in segments])
        best_segment_i = np.where(segment_area==np.max(segment_area))[0][0]
    
    json_kp = json_kps['kp-videos'][best_segment_i]
    filename = json_kp['path']
    imgs,fps = process_video(filename, args.skip_frame)
    filename = os.path.basename(filename)
    end =  int(len(imgs)) + end_cut
    # print(filename, end)
    imgs = imgs[start:end]
    
    os.makedirs(args.output,exist_ok=True)
    if filename.endswith("avi"):
        output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
    else:
        output_filename = os.path.join(args.output, filename) 
    segments = []
    segment_input = os.path.join(args.input, filename.replace('.mp4',f'-segments-{start}-{end}.mp4'))
    if os.path.exists(segment_input):
        segment_video, _ = process_video(segment_input, 1)
        segments = [cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) for seg in segment_video]
    else:
        segments = tool.kpimgs2segments(imgs, use_batch)
        segment_video = [cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) for seg in segments]
        imgs2video(segment_video, segment_input)
    
    trans_list, scores = segments2trans(segments, threshold, output_filename=output_filename, debug=debug)
        
    
    for i in range(len(json_kps['kp-videos'])):
        if video_i is not None:
            if i != video_i:
                continue
        json_kp = json_kps['kp-videos'][i]
        filename = json_kp['path']
        imgs,fps = process_video(filename, args.skip_frame)
        filename = os.path.basename(filename)
        end =  int(len(imgs)) + end_cut
        # print(filename, end)
        imgs = imgs[start:end]
        
        os.makedirs(args.output,exist_ok=True)
        if filename.endswith("avi"):
            output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
        else:
            output_filename = os.path.join(args.output, filename) 
        
        # ==========================Affine All Videos======================
        imgs = np.array(imgs)
        new_imgs = imgs2imgs_by_trans(imgs, trans_list, debug)
        imgs2video(new_imgs, output_filename, int(fps / args.skip_frame))


def video2video_by_phase(args, filename, start=0, end_cut = -1, threshold=0.2, use_batch=False, debug=False):
    """将视频做去抖

    Args:
        args (_type_): _description_
        video_i (int, optional): 若是None则全部做处理，否则指定一个视频做处理. Defaults to None.
        start (int, optional): _description_. Defaults to 0.
        end (int, optional): _description_. Defaults to -1.
        threshold (float, optional): _description_. Defaults to 0.2.
        debug (bool, optional): _description_. Defaults to False.
    """
    tool = NailfoldTools()
    imgs,fps = process_video(os.path.join(args.input,filename), args.skip_frame)
    filename = os.path.basename(filename)
    end =  int(len(imgs)) + end_cut
    # print(filename, end)
    imgs = imgs[start:end]
    
    os.makedirs(args.output,exist_ok=True)
    if filename.endswith("avi"):
        output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
    else:
        output_filename = os.path.join(args.output, filename) 
    segment_input = os.path.join(args.output, filename.replace('.mp4',f'-segments-{start}-{end}.mp4'))
    if os.path.exists(segment_input):
        segment_video, _ = process_video(segment_input, 1)
        segments = [cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) for seg in segment_video]
    else:
        segments = tool.imgs2segments(imgs)
        segment_video = [cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) for seg in segments]
        imgs2video(segment_video, segment_input)
    
    trans_list, scores = segments2trans(segments, threshold, output_filename=output_filename, debug=debug)
    
    # ==========================Affine All Videos======================
    imgs = np.array(imgs)
    new_imgs = imgs2imgs_by_trans(imgs, trans_list, debug)
    imgs2video(new_imgs, output_filename, int(fps / args.skip_frame))


    
def segments2trans(segments, threshold=0.2, output_filename=None, debug=False):
    """通过对分割结果做相位相关获得偏移量trans_list

    Args:
        segments (_type_): _description_
        start (int, optional): _description_. Defaults to 0.
        end (int, optional): _description_. Defaults to -1.
        threshold (float, optional): _description_. Defaults to 0.2.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: trans_list, scores
    """
    trans_list = []
    debug_video = []
    origin_video = []
    phases = []
    scores = []
    old_scores = []
    target = 0
    it = tqdm.tqdm(enumerate(segments))
    it.set_description("Phase Correlating")
    for i,seg in it:
        segment = cv2.cvtColor(seg,cv2.COLOR_GRAY2BGR)
        trans, result, new_seg = phase_affine(seg, segments[target], segment)
        visual_result = visualize_phase(result)
        score = check_result(segments[target], cv2.cvtColor(new_seg,cv2.COLOR_BGR2GRAY))
        old_score = check_result(segments[target], seg)
        scores.append(score)
        old_scores.append(old_score)
        trans_list.append(trans)
        # if score > threshold:
        #     print(i)
        #     break
        if debug:
            phases.append(visual_result)
            origin_video.append(segment/2 + cv2.cvtColor(segments[target],cv2.COLOR_GRAY2BGR)/2)
            debug_video.append(new_seg/2 + cv2.cvtColor(segments[target],cv2.COLOR_GRAY2BGR)/2)
    if debug:
        imgs2video(phases, output_filename.replace('.mp4',f'-result.mp4'))
        imgs2video(debug_video, output_filename.replace('.mp4',f'-new.mp4'))
        imgs2video(origin_video, output_filename.replace('.mp4',f'-origin.mp4'))
    plt.figure()
    scores= np.array(scores)
    old_scores = np.array(old_scores)
    plt.plot(np.arange(scores.shape[0]),scores,label="after",c='r')
    plt.plot(np.arange(old_scores.shape[0]),old_scores,label="before",c='g')
    plt.legend()
    plt.savefig(output_filename.replace('.mp4',f'-mse.png'))
    return trans_list, scores
    

if __name__=="__main__":
    args = parse_args()
    args.input = "videos/7.28/55896"
    args.output = "debug/55896"
    i = 0
    video2video_by_phase(args,filename="wmv2.mp4",use_batch=True, debug=True)
