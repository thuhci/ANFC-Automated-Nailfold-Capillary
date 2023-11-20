# count files num in given dir
import json
import os

dir = "/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/data/nailfold_dataset1"
# # dir = "/home/user/nailfold/zhaolx/Full_Pipeline/Data_Preprocess/data/classify_dataset"

mode_ls = os.listdir(dir)
for mode in mode_ls:
    dir_path = os.path.join(dir, mode)
    mode0_ls = os.listdir(dir_path)
    for mode0 in mode0_ls:
        dir_path = os.path.join(dir, mode, mode0)
        files = os.listdir(dir_path) # end up with .jpg
        files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
        print(f"{dir_path} files num {len(files)}")

# mode = "train/annotations"
# dir_path = os.path.join(dir, mode)
# files = os.listdir(dir_path)
# num = 0
# for file in files:
#     # read json file
#     file_path = os.path.join(dir_path, file)
#     with open(file_path, "r") as f:
#         dict = json.load(f)
#     num += len(dict['classes'])
# print(f"{dir_path} files num {num}")

