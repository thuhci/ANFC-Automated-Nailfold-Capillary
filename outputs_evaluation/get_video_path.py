# list file /home/user/nailfold/zhaolx/Full_Pipeline/Video_Process/nailfold_video_process/kp_videos/7.28/58497/wmv1-aligned/kp-0.mp4
import os

dir_ls = ["./Video_Process/data/kp_videos/7.28","./Video_Process/data/kp_videos/7.29"]
aligned_video_path_dict = {}
for dir in dir_ls:
    for root, dirs, files in os.walk(dir):
        if "-aligned" in root:
            for file in files:
                if file.endswith(".mp4") and file.startswith("kp"):
                    # print(os.path.join(root, file))
                    # aligned_video_path_ls.append(os.path.join(root, file))
                    name = root.split("/")[-2]
                    if name not in aligned_video_path_dict:
                        aligned_video_path_dict[name] = []
                    aligned_video_path_dict[name].append(os.path.join(root, file))

print(aligned_video_path_dict)
print(len(aligned_video_path_dict)) # 66
# write in json file
import json

with open("./outputs_evaluation/labels/aligned_video_path_dict.json", 'w') as f:
    json.dump(aligned_video_path_dict, f, indent=4)
    print(f"write in json file at {os.path.abspath('./outputs_evaluation/labels/aligned_video_path_dict.json')}")
# print(len(aligned_video_path_ls)) # 1023
# files in dir