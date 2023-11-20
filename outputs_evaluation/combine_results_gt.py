# load ./outputs_evaluation/labels/results_gt_individual.json
import json

import numpy as np

save_path = './outputs_evaluation/labels/results_gt_individual.json'
path_gt1 = './outputs_evaluation/labels/results_dict_gt_float.json'
path_gt2 = './outputs_evaluation/labels/results_dict_gt_bool.json'
path_pred = './outputs_evaluation/labels/results_pred.json'
with open(path_gt1, 'r') as f:
    gt1 = json.load(f)

with open(path_gt2, 'r') as f:
    gt2 = json.load(f)

with open(path_pred, 'r') as f:
    pred = json.load(f)

# print(gt1.keys())
gt_dict = {}
gt1_keys = list(gt1.keys())
pred_keys = list(pred.keys())
# pred_short_keys = [key.split('_')[1] for key in pred_keys]
for i in range(len(gt1_keys)):
    print(gt2[i])
    print(gt1_keys[i])
    if gt2[i]['index'] in gt1_keys[i]:
        keys_ls = ['visibility', 'num > 7', 'cross ratio <= 0.3', 'abnormal ratio <= 0.1']
        for key in keys_ls:
            gt1[gt1_keys[i]]['image_info'][key] = gt2[i][key]
        try:
            if gt1[gt1_keys[i]]['image_info']["output diameter"] < gt1[gt1_keys[i]]['image_info']["input diameter"]:
                # switch
                gt1[gt1_keys[i]]['image_info']["output diameter"], gt1[gt1_keys[i]]['image_info']["input diameter"] = gt1[gt1_keys[i]]['image_info']["input diameter"], gt1[gt1_keys[i]]['image_info']["output diameter"]
            gt1[gt1_keys[i]]['image_info']["output input ratio"] = round(gt1[gt1_keys[i]]['image_info']["output diameter"] / gt1[gt1_keys[i]]['image_info']["input diameter"],2)
        except:
            raise InterruptedError
            # gt1[gt1_keys[i]]['image_info']["output input ratio"] = -1
    
        gt_dict = gt1
        # replace the img name
        # for pred_key in pred_keys:
        #     if gt1_keys[i] in pred_key:
        #         gt_dict[pred_keys[i]] = gt1[gt1_keys[i]]
        #         break


    else:
        print('Error: index not match')
        break

# save gt1
with open(save_path, "w") as f:
    json.dump(gt_dict, f, indent=4)
    print("Save to {}".format(save_path))

