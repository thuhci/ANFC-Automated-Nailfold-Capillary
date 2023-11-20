import json
import os

from Image_Analysis.nailfold_image_profile.image_analysis import image_analysis
from Video_Process.nailfold_video_process.estimate_flow import *
from Video_Process.nailfold_video_process.phase_correlation import *
from Video_Process.nailfold_video_process.tools import NailfoldTools
from Video_Process.nailfold_video_process.utils.process_video import *
from Video_Process.nailfold_video_process.video2keypoint import *
from Video_Process.nailfold_video_process.video2video import *

"""
将一个视频或一个文件夹的视频，分析其血管指标，输出为json
--input: input file or input dir
--output: output dir
"""

def process_kp_video(input, i):
    """处理一个单一血管为中心的裁剪视频，输出该血管的特征

    Args:
        input (str): align_path
    """
    args.input = input
    args.output = input
    try:
        data = estimate_flow(args, i, [5], visualize=True)
    except (Exception) as e:
        print(f"[Error]: flow estimate error in {input}: {e}")
        return None
    
    return data


def process_file(input,output):
    id = os.path.basename(os.path.dirname(input))
    name = id +"_"+ os.path.basename(input).replace("mp4","json")
    cache_output = os.path.join(output, "cache")
    video_info = video_process_file(input, cache_output)
    photo_info = photo_process_file(input, cache_output)
    info = {
        'path': input,
        'video_info': video_info,
        'photo_info': photo_info,
    }
    with open(os.path.join(output, name),"w") as f:
        json.dump(info, f, indent=4)
    return


def photo_process_file(input, output):
    os.makedirs(output,exist_ok=True)
    imgs, fps = process_video(input, 1)
    image = imgs[0]
    image_data = image_analysis([image], ["class_"+os.path.basename(input).replace("mp4", "png")], output_dir=output)
    bboxes = image_data['bboxes'][0]
    classes = image_data['classes'][0]
    photo_info = {
        'num': len(classes),
        'cross_num': classes.count('cross'),
        'abnormal_num': classes.count('abnormal'),
        'normal_num': classes.count('normal'),
        'blur_num': classes.count('blur')
    }
    
    return photo_info


def video_process_file(input, output):
    os.makedirs(output,exist_ok=True)
    filename = os.path.basename(input)
    # wmv1.mp4
    dirname = filename.replace(".mp4","")
    args.input = os.path.dirname(input)
    args.output = output
    video2video_by_detect_and_phase(args,filename)

    # /wmv1
    kp_path = os.path.join(output, dirname)
    args.input = os.path.join(output, filename)
    args.output = kp_path
    if not os.path.exists(kp_path):
        video2keypoint(args)

    # refine align
    align_dirname = dirname + "-aligned"
    align_path = os.path.join(output, align_dirname)
    os.makedirs(align_path,exist_ok=True)

    with open(os.path.join(kp_path, "kps.json"), "r") as f:
        json_kps = json.load(f)
    for i,kp in enumerate(json_kps['kp-videos']):
        # kp-1.mp4
        kp_name = os.path.basename(kp['path'])
        json_kps['kp-videos'][i]['path'] = os.path.join(align_path ,kp_name)
    with open(os.path.join(align_path, "kps.json"), "w") as f:
        json.dump(json_kps, f, indent=4)
        
    args.input = kp_path
    args.output = align_path
    error = kpvideo2kpvideo_by_phase(args, use_batch=True)
    if error is not None:
        return
    info = []

    # [最小、平均、最大]
    velocity_info = []
    white_cell_info = []
        
    for i in range(len(json_kps['kp-videos'])):
        data = process_kp_video(align_path, i)
        if data is None:
            continue
        info.append(data)

        if 'velocity' in data:
            velocity_info.append(data['velocity'])
            white_cell_info.append(data['white_cell'])
        if len(velocity_info)>3:
            break
    
    velocity_info = np.array(velocity_info)
    velocity_info = np.abs(velocity_info)
    white_cell_info = np.array(white_cell_info)

    top_diameter_info = np.array([data['top_diameter'] for data in info])
    input_diameter_info = np.array([data['input_diameter'] for data in info])
    output_diameter_info = np.array([data['output_diameter'] for data in info])
    length_info = np.array([data['length'] for data in info])
    video_info = {
        'velocity': velocity_info.mean(0,keepdims=True).tolist(),
        'white_cell': [
            int(white_cell_info.min()), 
            int(white_cell_info.mean()), 
            int(white_cell_info.max())
            ],
        'top_diameter': top_diameter_info.mean(),
        'input_diameter': input_diameter_info.mean(),
        'output_diameter': output_diameter_info.mean(),
        'length': length_info.mean()
    }
    return video_info


if __name__ == '__main__':
    args = parse_args()
    # 看7.29中的所有文件夹
    args.input = "/home/user/nailfold/20221003_nailfold/video/59791"
    args.output = "./debug/pipeline/59791"
    root = args.input
    output = args.output
    os.makedirs(output,exist_ok=True)
    if os.path.isfile(root):
        process_file(root, output)
    else:
        for file in os.listdir(root):
            if file.startswith("."):
                continue
            if not file.endswith(".mp4"):
                continue
            if file.endswith("kp.mp4"):
                continue
            process_file(os.path.join(root,file), output)