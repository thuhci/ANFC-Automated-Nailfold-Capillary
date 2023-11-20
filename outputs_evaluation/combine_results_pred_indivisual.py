
import argparse
import json
import os

import numpy as np


def dict_add(dict1, dict2):
    common_keys = [key for key in dict1.keys() if key in dict2.keys()]
    for key in common_keys:
        dict1[key] += dict2[key] if dict2[key] != -1 else 0
    return dict1

def dict_bool(dict1, dict2):
    common_keys = [key for key in dict1.keys() if key in dict2.keys()]
    for key in common_keys:
        dict1[key] += dict2[key] if dict2[key] != -1 else 0
    return dict1

def dict2array(dict1, dict2):
    ls = []
    common_keys = [key for key in dict1.keys() if key in dict2.keys()]
    for key in common_keys:
        ls.append(dict2[key])
    return np.array(ls)

def array2dict(dict1, array):
    keys = list(dict1.keys())
    assert len(keys) == len(array)
    for i in range(len(array)):
        if array[i] == True or array[i] == False:
            dict1[keys[i]] = int(array[i])
        else:
            dict1[keys[i]] = array[i]
    return dict1


def individual_mean(indivisual_idx, pred):
    indivisual_imgs = [key for key in pred_keys if key.split('_')[1] == str(indivisual_idx)]
    valid_indivisual_imgs = [img for img in indivisual_imgs if pred[img]["image_info"]["visibility"]]

    info_dict_float = {"num per image":0, "output diameter": 0, "input diameter": 0, "top_diameter": 0, "length": 0, "num": 0, "normal num": 0, "cross num": 0, "abnormal num": 0}
    info_dict_bool = {"cross ratio <= 0.3": 0, "abnormal ratio <= 0.1": 0, "visibility": len(valid_indivisual_imgs)>0} # "num > 7": 0
    info_array_float = np.zeros([len(valid_indivisual_imgs), len(info_dict_float.keys())])
    info_array_bool = np.zeros([len(valid_indivisual_imgs), len(info_dict_bool.keys())])
    
    for i, img in enumerate(valid_indivisual_imgs):
        info_array_float[i] = dict2array(info_dict_float, pred[img]['image_info'])
        info_array_bool[i] = dict2array(info_dict_bool, pred[img]['image_info'])

    info_array_float_mean = np.nanmean(info_array_float, axis=0)
    info_array_float_mean = np.round(info_array_float_mean, 3)

    # BUG: should not add bool directly
    info_array_bool_and = np.nanmean(info_array_bool, axis=0) > 0.5 #== 1
    # print(info_array_float_mean, info_array_bool_and)

    info_dict_float = array2dict(info_dict_float, info_array_float_mean)
    info_dict_bool = array2dict(info_dict_bool, info_array_bool_and)

    info_dict = info_dict_bool
    for k,v in info_dict_float.items():
        info_dict[k] = v

    info_dict["num > 7"] = 1 if info_dict["num per image"] > 7 else 0
    info_dict["cross ratio <= 0.3"] = int(np.round((info_dict["cross num"] / info_dict["num"]),1) <=0.3)
    info_dict["abnormal ratio <= 0.1"] = int(np.round((info_dict["abnormal num"] / info_dict["num"]),1) <=0.1)
    info_dict["input output ratio"] = info_dict["output diameter"] / info_dict["input diameter"]
    info_dict["input output ratio"] = np.round(info_dict["input output ratio"], 2)

    return info_dict
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_org_dir', type=str, default='./outputs_evaluation/labels')
    parser.add_argument('--pred_org_file', type=str, default='results_pred.json')
    parser.add_argument('--save_dir', type=str, default='./outputs_evaluation/labels')
    parser.add_argument('--save_file', type=str, default='results_pred_individual.json')
    args = parser.parse_args()

    path_pred = os.path.join(args.pred_org_dir, args.pred_org_file)
    save_path = os.path.join(args.save_dir, args.save_file)

    with open(path_pred, 'r') as f:
        pred = json.load(f)

    pred_keys = list(pred.keys())
    pred_individual_keys = [key.split('_')[1] for key in pred_keys]
    pred_individual_keys.sort()

    

    # indivisual_idx = '55868'
    pred_dict = {}
    for indivisual_idx in pred_individual_keys:
        info_dict = individual_mean(indivisual_idx, pred)
        pred_dict[indivisual_idx] = {'image_info': info_dict}

    with open(save_path, "w") as f:
        json.dump(pred_dict, f, indent=4)
        print("Save to {}".format(save_path))