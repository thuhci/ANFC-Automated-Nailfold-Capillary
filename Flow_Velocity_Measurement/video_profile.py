


import argparse
import json
import math
import os
import typing

import numpy as np
from Flow_Velocity_Measurement.utils.utils_maxima import (filter, find_extrema,
                                                          vote)
from Flow_Velocity_Measurement.utils.utils_morph import ImageSegmentation
from Flow_Velocity_Measurement.utils.utils_visualization import (
    draw_boxplot, draw_boxplot_and_annotate, draw_profiles, draw_scatter,
    draw_wbc)
from Flow_Velocity_Measurement.video_seg import get_video_best_seg
from skimage.transform import iradon, radon
from Video_Process.nailfold_video_process.utils.process_video import \
    process_video


def t_video_analysis(video_path, output_dir, pos: tuple, visualize: bool = False, split_num: int = 2, pad_ratio: float= 1, video_name = "video_profile")->typing.List[float]:
    '''
    This function is used to analyze the video and return the velocity of the white blood cell.
    Input:
        video_path: the path of the video
        output_dir: the path to save the results
        pos: the position of seed to point out the artery to be analyzed
        visualize: whether to visualize the results and save the figures
        split_num: the number of splits to divide the video frame to get higher resolution of segmentation
        pad_ratio: the ratio of padding to the original video frame to adapt the size of nailfold to training data
    Output:
        velocity_ums: the velocity of detected white blood cells
    '''    
    profiles,fps = get_profiles(video_path, pos, split_num=split_num, pad_ratio=pad_ratio)
    
    sinogram = radon_test(profiles)

    extremas = find_extrema(sinogram)
    # extremas = vote(sinogram, extremas)
    extremas = filter(sinogram, extremas)

    degrees = [extrema[1]/sinogram.shape[1]*180 for extrema in extremas]
    velocity = [math.tan(degree*math.pi/180) for degree in degrees] # pixel/frame
    velocity_ums = [v*1.15*fps for v in velocity] # 1 pixel ~ 1.1 um, 1 frame ~ 0.5 s(fps=20), so 1 pixel/frame ~ 22 um/s
    print(f"-----RESULTS----- \n Degrees:{degrees} \nVelocity(/ums){velocity_ums}")

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Save visualized results at {output_dir}")
        
        draw_profiles(profiles, output_dir, f"{video_name}_profiles.png")
        draw_scatter(sinogram, extremas, output_dir, f"{video_name}_scatter.png")
        
        if len(extremas):
            extremas_field = np.zeros_like(sinogram)
            for extrema in extremas:
                extremas_field[round(extrema[0])][round(extrema[1])] = 1
            reconstruction_fbp = iradon(extremas_field)
            wbc_time = count_time(reconstruction_fbp)
            
            draw_wbc(wbc_time, velocity_ums, profiles.shape[0], fps, output_dir, f"{video_name}_WBC_events.png")    
            draw_boxplot_and_annotate(velocity_ums, output_dir, f"{video_name}_boxplot.png")
    
    return velocity_ums


def get_profiles(video_path, pos=(132, 70), skip_frame=1, split_num=2, pad_ratio=1):
    
    imgs, fps = process_video(video_path, skip_frame)
    frame_num = len(imgs) 
    # print(f"Seed at {pos}")
    # print(f"Frame length {frame_num}, fps {fps}")
    # print(f'Frame size{imgs[0].shape}')
    
    best_seg = get_video_best_seg(
        images=imgs, split_num=split_num, pad_ratio=pad_ratio)
    myseg = ImageSegmentation(best_seg)
    myseg.save_seg("/home/user/nailfold/zhaolx/Full_Pipeline/Flow_Velocity_Measurement/output_seg", video_path.split("/")[-3])
    
    seed = myseg.get_nearest_seed(myseg.img_thinning, pos)
    
    if seed is None:
        return [], fps
    conncomp,endpoints,junctions = myseg.get_conncomp(myseg.img_thinning, seed)

    print(f"Original nailfold length {len(conncomp)}")
    conncomp = conncomp[len(conncomp)//5:5*len(conncomp)//7]
    
    if len(conncomp) < 100: # threshold
        return [], fps
    
    profiles = []
    for _, img in enumerate(imgs[:]):
        profiles.append(myseg.get_profile(img.astype(np.uint8), conncomp).reshape([1,-1]))
    profiles = np.concatenate(profiles, axis=0)
    return profiles, fps


def radon_test(profiles):
    if len(profiles) == 0 or max(profiles.shape) < 120:
        return []
    
    theta = np.linspace(0., 180., max(profiles.shape), endpoint=False)
    sinogram = radon(profiles, theta=theta)
    M = np.max(np.max(sinogram))
    m = np.min(np.min(sinogram))
    sinogram = (sinogram*2 - M - m)/(M - m)
    sinogram = np.flipud(sinogram)
    return sinogram


def count_time(reconstruction_fbp):
    wbc_time = np.where(reconstruction_fbp[:,reconstruction_fbp.shape[1]//2]>0.001)[0]
    print(f"WBC time {wbc_time}")
    return wbc_time


def t_video_analysis_subject(subject_name, video_path_ls, config):
    print(f"Analyze subject {subject_name}...")
    wbc_events = []
    for video_path in video_path_ls:
        video_name = subject_name + '-'+video_path.split("/")[-2].split('-')[0][-1] +'-'+ video_path.split("/")[-1].split('-')[1][0] 
        output_dir = os.path.join(config.output_dir, subject_name)
        try:
            velocity_ums = t_video_analysis(video_path, output_dir, video_name=video_name, pos=(config.nailfold_pos_x, config.nailfold_pos_y), visualize=config.visualize, split_num=config.split_num, pad_ratio=config.pad_ratio)
        except:
            print("error!")
            velocity_ums = []
        wbc_events.extend(velocity_ums)
    
    os.makedirs(output_dir, exist_ok=True)
    draw_boxplot(wbc_events, config.output_dir, f"{subject_name}_boxplot.png")
    
    wbc_events = np.round(np.array(wbc_events),3)
    median_velocity = np.nanmedian(wbc_events)
    mean_velocity = np.nanmean(wbc_events)
    wbc_events_dict = {subject_name:{"velocity": median_velocity, "mean_velocity": mean_velocity, "wbc_events": wbc_events.tolist()}}
    print(f"Median velocity of {subject_name} is {median_velocity}.")
    
    # save to json
    with open(os.path.join(config.output_dir, f"results_velocity_individual.json"), 'a') as f:
        json.dump(wbc_events_dict, f, indent=4)
    return wbc_events


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path_dict_file', type=str, default="")
    parser.add_argument('--video_name', type=str, default="")
    parser.add_argument('--video_type', type=str, default=".mp4")
    parser.add_argument('--video_path', type=str, default="./Flow_Velocity_Measurement/video_sample")
    parser.add_argument('--output_dir', type=str, default="./Flow_Velocity_Measurement/output_sample/")
    
    # set seed to point out which artery is to be analyzed
    parser.add_argument('--nailfold_pos_x', type=int, default=150)
    parser.add_argument('--nailfold_pos_y', type=int, default=100)

    # Dividing into patches or padding so that the size of the nailfold matches the training data resolution
    parser.add_argument('--split_num', type=int, default=1) # patches number = split_num * split_num
    parser.add_argument('--pad_ratio', type=int, default=1) # pad_ratio >= 1
    
    # whether to save the visualized results
    parser.add_argument('--visualize', action="store_true")

    config = parser.parse_args()
    
    if config.video_path_dict_file != "":
        # eval subjects in video_path_dict_file
        with open(config.video_path_dict_file, 'r') as f:
            video_path_dict = json.load(f)
        
        if config.video_name != "":
            # a)eval given subject
            subject_name = config.video_name
            video_path_ls = video_path_dict[subject_name]
            t_video_analysis_subject(subject_name, video_path_ls, config)
        else:
            # b)eval all subjects
            for subject_name, video_path_ls in video_path_dict.items():
                t_video_analysis_subject(subject_name, video_path_ls, config)
    else:
        # c)eval given video
        video_path = os.path.join(config.video_path,config.video_name + config.video_type)
        subject_name = config.video_name
        t_video_analysis_subject(subject_name, [video_path], config)
