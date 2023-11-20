from Video_Process.nailfold_video_process.utils.process_video import *
import numpy as np
import cv2

args = parse_args()
root = args.input
output = args.output
os.makedirs(output, exist_ok=True)
print(len(os.listdir(root)))
for dir in os.listdir(root):
    # /59894
    if (dir.startswith(".")):
            continue
    dir_path = os.path.join(root, dir)
    if not os.path.isdir(dir_path):
            continue
    for video in os.listdir(dir_path):
        if (video.startswith("._")):
            continue
        video_path = os.path.join(dir_path, video)
        imgs, fps = process_video(video_path, 100)
        name = video.replace("mp4","png")
        name = f"20221003_{dir}_{name}"
        cv2.imwrite(os.path.join(output, name), imgs[0])