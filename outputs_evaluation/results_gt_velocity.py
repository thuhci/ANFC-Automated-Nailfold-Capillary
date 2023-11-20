import json
import os

data_dir_ls = ["/home/user/nailfold/Tangshan-Samples/7.28", "/home/user/nailfold/Tangshan-Samples/7.29"]
files_name_ls = ["REP.fmt", "REP.fmt"]
all_results = {}
for data_dir,file_name in zip(data_dir_ls,files_name_ls):
    dir_names = os.listdir(data_dir)
    for dir in dir_names:
        file = os.path.join(data_dir, dir, file_name)
        # read 1722 line in REP.fmt
        with open(file, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
            # 'utf-8' codec can't decode byte 0xc1 in position 178: invalid start byte

        line = lines[1721] # delete \n
        line = line.split("\n")[0]
        all_results[dir] = int(line)
        
        

# save all_results to json
save_dir = "./outputs_evaluation/labels"
with open(os.path.join(save_dir, "results_dict_gt_velocity.json"), 'w') as f:
    json.dump(all_results, f, indent=4)