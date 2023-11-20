import json
import os

from Video_Process.nailfold_video_process.estimate_flow import *
from Video_Process.nailfold_video_process.phase_correlation import *
from Video_Process.nailfold_video_process.utils.process_video import *
from Video_Process.nailfold_video_process.video2keypoint import *


def write_log(str, output):
    with open(os.path.join(output,"list.txt"),"a") as file:
        file.write(str+"\n")

if __name__ == '__main__':
    args = parse_args()
    # args.input = "./Video_Process/data/aligned_videos/7.28"
    # args.output = "./kp_videos/7.28"
    # 看7.29中的所有文件夹
    root = args.input
    output = args.output
    os.makedirs(output,exist_ok=True)
    list = os.listdir(root)
    already_list = os.listdir(output)
    for dir in list:
        # /49510
        if dir in already_list:
            continue
        write_log("-"+dir, output)
        video_dir_path = os.path.join(root,dir)
        output_dir = os.path.join(output,dir)
        if not os.path.isdir(video_dir_path):
            continue
        for file in os.listdir(video_dir_path):
            if not file.endswith(".mp4"):
                continue
            if file.endswith("kp.mp4"):
                continue
            # wmv1.mp4
            dirname = file.replace(".mp4","")
            write_log("--"+dirname, output)
            # /wmv1
            kp_path = os.path.join(output_dir,dirname)
            args.input = os.path.join(video_dir_path, file)
            args.output = kp_path
            if not os.path.exists(kp_path):
                video2keypoint(args)

            # refine align
            align_dirname = dirname + "-aligned"
            align_path = os.path.join(output_dir,align_dirname)
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
                continue
            
            for i in range(len(json_kps['kp-videos'])):
                args.input = align_path
                args.output = align_path
                try:
                    all_flow = estimate_flow(args, i, [5,15])
                    write_log(f"----{len(all_flow)}", output)
                except Exception as e:
                    print("=======================",e,"=======================")