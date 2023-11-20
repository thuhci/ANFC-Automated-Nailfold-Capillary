import os
from Video_Process.nailfold_video_process.utils.process_video import *
from Video_Process.nailfold_video_process.video2video import *

def stablize_all(args):
    # 看7.29中的所有文件夹
    root = args.input
    output = args.output
    os.makedirs(output,exist_ok=True)
    for dir in os.listdir(root):
        print(dir)
        if dir in os.listdir(output):
            continue
        args.input = os.path.join(root,dir)
        args.output = os.path.join(output,dir)
        for filename in os.listdir(args.input):
            if not (filename.endswith("mp4") or filename.endswith("avi")):
                continue
            os.makedirs(args.output,exist_ok=True)
            video2video_by_detect(args,filename)

def save2mp4_all(args):
    # 看7.29中的所有文件夹
    root = args.input
    output = args.output
    os.makedirs(output,exist_ok=True)
    for dir in os.listdir(root):
        print(dir)
        if dir in os.listdir(output):
            continue
        args.input = os.path.join(root,dir)
        args.output = os.path.join(output,dir)
        for filename in os.listdir(args.input):
            os.makedirs(args.output,exist_ok=True)
            if not (filename.endswith("mp4") or filename.endswith("avi")):
                continue
            avi2mp4(args,filename)

if __name__ == '__main__':
    args = parse_args()
    # save2mp4_all(args)
    if args.type == "avi":
        save2mp4_all(args)
    elif args.type == "stable":
        stablize_all(args)
    else:
        print("=========Please specify type: avi/stable. =========")