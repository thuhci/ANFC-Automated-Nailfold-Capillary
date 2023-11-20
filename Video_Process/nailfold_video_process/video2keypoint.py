from Video_Process.nailfold_video_process.utils.process_video import *
import cv2
import os
import sys
import tqdm
import json
from Video_Process.nailfold_video_process.tools import NailfoldTools

def video2keypoint(args):
    os.makedirs(args.output,exist_ok=True)

    imgs,fps = process_video(args.input, 1)
    sample = imgs[0].copy()
    h,w,_ = sample.shape
    tool = NailfoldTools()
    
    keypoints, bboxes = tool.img2keypoint(sample)
    mask = tool.img2segment(sample)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    tool.visualize_keypoint("sample", cv2.cvtColor(imgs[0],cv2.COLOR_BGR2RGB), bboxes, keypoints, output_dir=args.output)
    cv2.imwrite(os.path.join(args.output,"mask.png"),mask)
    binarymask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY).astype(np.bool8)
    remove = []
    # 筛选方法是看其包围盒上边沿是否有mask
    for id,bbox in enumerate(bboxes):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        upper_bound = mask[min(bbox[1],bbox[3]),bbox[0]:bbox[2]]
        if np.any(upper_bound):
            remove.append(id)
    bboxes = np.delete(bboxes, remove, axis=0)
    keypoints = np.delete(keypoints, remove, axis=0)
    # scores = np.delete(scores, remove, axis=0)
            
    tool.visualize_keypoint("refine", cv2.cvtColor(imgs[0],cv2.COLOR_BGR2RGB), bboxes, keypoints, output_dir=args.output)

    
    # 拆分原画面
    json_record = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    PADDING = 100
    videos = []
    for id,kps in enumerate(keypoints.tolist()):
        file_path = os.path.join(args.output, f"kp-{id}.mp4")
        # if kps[0][1]-PADDING < 0 or kps[0][1]+2*PADDING > h or kps[0][0]-PADDING < 0 or kps[0][0]+PADDING > w:
        #     continue
        padding_x = [min(PADDING, kps[0][0]), min(PADDING,w - kps[0][0])]
        padding_y = [min(PADDING, kps[0][1]), min(2*PADDING,h - kps[0][1])]
        print(f"padding: {padding_x}, {padding_y}")
        # 新窗口内的关键点坐标
        new_kps = [(kp[0]-kps[0][0]+padding_x[0],kp[1]-kps[0][1]+padding_y[0]) for kp in kps]
        # 拆分视频
        imgs = np.array(imgs)
        # new_imgs = np.zeros([imgs.shape[0], 300, 200, 3], np.uint8)
        # [:, :padding_y[0]+padding_y[1], :padding_x[0]+padding_x[1]]
        new_imgs = imgs[:,kps[0][1]-padding_y[0]:kps[0][1]+padding_y[1],kps[0][0]-padding_x[0]:kps[0][0]+padding_x[1]]
        imgs2video(new_imgs,file_path,fps) 
        videos.append({
            'original_kp':kps,
            'kp':new_kps,
            'path':file_path
        })
    json_record['kp-videos'] = videos
    with open(os.path.join(args.output,"kps.json"),"w") as f:
        json.dump(json_record,f,indent=4)

if __name__ == '__main__':
    args = parse_args()
    args.input = "./debug/pipeline/cache/wmv0.mp4"
    args.output = "./debug/pipeline/cache/wmv0"
    video2keypoint(args)
    exit(0)