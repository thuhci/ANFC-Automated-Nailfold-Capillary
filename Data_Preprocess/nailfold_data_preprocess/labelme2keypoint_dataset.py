import json
import os
from utils.get_cfg import get_cfg

cfg_dict = get_cfg("keypoint")
dataset_dir = cfg_dict.get("dataset_dir")
keypoints_classes_ids2names = cfg_dict.get("keypoints_classes_ids2names")
class_name_dict = cfg_dict.get("class_name_dict")
CROSS_PADDING = cfg_dict.get("CROSS_PADDING")
TOP_PADDING = cfg_dict.get("TOP_PADDING")

H = 768
W = 1024

keypoints_classes = [keypoints_classes_ids2names[key] for key in keypoints_classes_ids2names]

conj_classes = []
only_cross = False
if 'conj' in class_name_dict:
    conj_classes = ['conj-1', 'conj-2', 'conj-3']
    if len(keypoints_classes_ids2names) == 1:
        only_cross = True

def parse_one(img_path, lab_path, output_folder):
    # 标注json路径
    print(lab_path)
    res = {}
    res["bboxes"] = []
    res["keypoints"] = []
    res["classes"] = []
    map = {}
    # PADDING = 50
    with open(lab_path, 'r') as f:
        data = json.load(f)
        for item in data['shapes']:
            if item["group_id"] in map:
                map[item["group_id"]].append(item.copy())
            else:
                map[item["group_id"]] = [item.copy()]

        for id in map.keys():
            # 共多于8个关键点
            if len(map[id]) < 8:
                continue
            if id is None: 
                continue
            all_points = {}
            # { 'up': [x, y, 1] }
            fail = 0
            xmin, ymin = W-1, H-1
            xmax, ymax = 0, 0
            for item in map[id]:
                x = int(item['points'][0][0])
                y = int(item['points'][0][1])
                if item['label'] in keypoints_classes:
                    xmin = min(xmin, max(0, x - TOP_PADDING))
                    xmax = max(xmax, min(W-1, x + TOP_PADDING))
                    ymin = min(ymin, max(0, y - TOP_PADDING))
                    ymax = max(ymax, min(H-1, y + TOP_PADDING))

                if item['label'] in conj_classes:
                    visible_keypoints = [[x,y,0] for _ in range(len(keypoints_classes_ids2names))]
                    # 令第一个up点为x,y,0
                    visible_keypoints[0][-1] = 1
                    res["keypoints"].append(visible_keypoints)
                    res["bboxes"].append([max(0, x - CROSS_PADDING), max(0, y - CROSS_PADDING), min(W-1, x + CROSS_PADDING), min(H-1, y + CROSS_PADDING)])
                    res['classes'].append(class_name_dict['conj'])
                
                if item['label'] in all_points:
                    fail = 1
                    break
                all_points[item['label']] = [x, y, 1]
            
            keypoints = []
            class_name = 'normal'
            for key in keypoints_classes_ids2names:
                if keypoints_classes_ids2names[key] not in all_points:
                    if keypoints_classes_ids2names[key] == 'conj-1':
                        class_name = 'normal'
                        keypoints.append(all_points['up'])
                    elif keypoints_classes_ids2names[key] == 'conj-2':
                        if class_name == 'conj-2':
                            class_name = 'conj-1'
                        keypoints.append(all_points['up'])
                    else:
                        fail = 1
                        break
                    continue
                keypoints.append(all_points[keypoints_classes_ids2names[key]])
            
            # duplicate keys in label
            if fail==1:
                continue
            if not only_cross:
                res["keypoints"].append(keypoints)
                res["bboxes"].append([xmin, ymin, xmax, ymax])
                res['classes'].append(class_name_dict[class_name])
    # print(res)
    if len(res['classes']) == 0:
        return 1
    assert(len(res["keypoints"]) == len(res["bboxes"]))
    img_name = lab_path.split('/')[-1]
    output_dir = os.path.join(output_folder, img_name)
    res_d = json.dumps(res,indent=4)
    with open(output_dir, 'w') as f: 
        f.write(res_d)
    return 0


def main():
    raw_dir = '../data/tangshan'
    # dataset_dir = './data/keypoint_dataset'
    output_folder = os.path.join(dataset_dir, 'train/annotations')
    image_folder = os.path.join(dataset_dir, 'train/images')
    test_output_folder = os.path.join(dataset_dir, 'test/annotations')
    test_image_folder = os.path.join(dataset_dir, 'test/images')
    imgs_path = raw_dir
    labs_path = raw_dir
    cnt = 0
    files = []
    for file in os.listdir(imgs_path):
        if file.endswith(".json"):
            continue
        files.append(file)
    files.sort()
    for file in files:
        cnt += 1
        img_path = os.path.join(imgs_path, file)
        lab_path = os.path.join(labs_path, file.replace('.png', '.json').replace('.jpg', '.json'))
        if cnt > len(files) * 2 / 3:
            output_dir = test_output_folder
            image_dir = test_image_folder
        else:
            output_dir = output_folder
            image_dir = image_folder
        os.makedirs(output_dir,exist_ok=True)
        os.makedirs(image_dir,exist_ok=True)
        
        error = parse_one(img_path, lab_path, output_dir)
        if error:
            continue
        os.system(f'cp {img_path} {image_dir}/{file}')
        # os.system(f'mv {json_path} {new_json_path}')
        print(cnt)


main()