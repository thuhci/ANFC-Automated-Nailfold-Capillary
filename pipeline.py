import os
import json
from Video_Process.nailfold_video_process.utils.process_video import *
from Video_Process.nailfold_video_process.video2keypoint import *
from Video_Process.nailfold_video_process.estimate_flow import *
from Video_Process.nailfold_video_process.phase_correlation import *
from Video_Process.nailfold_video_process.video2video import *
from Video_Process.nailfold_video_process.tools import NailfoldTools

"""
将一个视频或一个文件夹的视频，分析其血管指标，输出为json
--input: input file or input dir
--output: output dir
"""


def process_file(input,output):
    name = input.replace("./","").replace("/","_").replace("mp4","json")
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
    imgs, fps = process_video(input, 1)
    image = imgs[0]
    tool = NailfoldTools()
    keypoints, bboxes = tool.imgs2keypoint([image])
    keypoints = keypoints[0]
    bboxes = bboxes[0]
    # 检测数量
    num = len(bboxes)
    keypoints = np.array(keypoints)
    top_diameter = np.linalg.norm(keypoints[:,0] - keypoints[:,1], axis=-1)
    left_diameter = np.linalg.norm(keypoints[:,2] - keypoints[:,3], axis=-1)
    right_diameter = np.linalg.norm(keypoints[:,4] - keypoints[:,5], axis=-1)
    photo_info = {
        'num': num,
        'top_diameter': [top_diameter.min(), top_diameter.mean(), top_diameter.max()],
        'left_diameter': [left_diameter.min(), left_diameter.mean(), left_diameter.max()],
        'right_diameter': [right_diameter.min(), right_diameter.mean(), right_diameter.max()]
    }
    return photo_info


def video_process_file(input, output):
    os.makedirs(output,exist_ok=True)
    filename = os.path.basename(input)
    # wmv1.mp4
    dirname = filename.replace(".mp4","")
    args.input = os.path.dirname(input)
    args.output = output
    video2video_by_detect(args,filename)

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
    for i in range(len(json_kps['kp-videos'])):
        args.input = align_path
        args.output = align_path
        try:
            all_flow = estimate_flow(args, i, [5])
            if len(all_flow) > 2:
                data = flow2data(all_flow)
                info.append(data)
                if len(info)>3:
                    break
        except Exception as e:
            print("=======================",e,"=======================")
    # [最小、平均、最大]
    velocity_info = np.array([data['velocity'] for data in info])
    velocity_info = np.abs(velocity_info)
    video_info = {
        'velocity': velocity_info.mean(0,keepdims=True).tolist()
    }
    return video_info


if __name__ == '__main__':
    args = parse_args()
    # 看7.29中的所有文件夹
    args.input = "/home/user/nailfold/20221003_nailfold/video/54781"
    args.output = "./test"
    root = args.input
    output = args.output
    os.makedirs(output,exist_ok=True)
    if os.path.isfile(root):
        process_file(root, output)
    else:
        for file in os.listdir(root):
            if not file.endswith(".mp4"):
                continue
            if file.endswith("kp.mp4"):
                continue
            process_file(os.path.join(root,file), output)