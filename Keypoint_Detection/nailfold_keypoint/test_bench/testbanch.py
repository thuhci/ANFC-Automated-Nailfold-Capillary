import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Keypoint_Detection.nailfold_keypoint.dataset import (KeypointDataset,
                                                          collate_fn,
                                                          kp_train_transform)
from torch.utils.data import DataLoader

if __name__ == "__main__":

    dataset_path = "./Keypoint_Detection/data/nailfold_dataset1/train"
    def get_keypoint_gt_diameter(dataset_path):

        keypoints_num = 8 # for conj dataset, 1; for nailfold keypoint dataset 8
        
        dataset = KeypointDataset(
            dataset_path, keypoint_num=keypoints_num, transform=kp_train_transform(), demo=True)
        data_loader = DataLoader(dataset, batch_size=1,
                                shuffle=False, collate_fn=collate_fn)

        iterator = iter(data_loader)

        keypoints_pair_dict = {}
        # iterate batch
        for batch in iterator:
            # image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bboxes, keypoints, labels = dataset.target2numpy(batch[1])

            def dist(kp1, kp2):
                # import math
                # return math.sqrt((kp1[0]-kp2[0])**2 + (kp1[1]-kp2[1])**2)
                return np.linalg.norm(kp1 - kp2, axis=-1)
            
            keypoints_pair = {'input_diameter':[],'output_diameter':[],'top_diameter':[],'length':[],'left_keypoints':[],'right_keypoints':[],'top_keypoints':[]}
            for kp_group in keypoints:
                
                kp_group = np.array(kp_group)
                centers = [(round((kp_group[i][0]+kp_group[i+1][0])/2),round((kp_group[i][1]+kp_group[i+1][1])/2)) for i in range(0, len(kp_group)-2, 2)]
                # round centers
                dists = [dist(kp_group[i], kp_group[i+1]) for i in range(0, len(kp_group)-2, 2)]
                
                left_idx = np.argmin([kp[0] for kp in centers])
                right_idx = np.argmax([kp[0] for kp in centers])
                top_idx = np.argmin([kp[1] for kp in centers])
                
                if dists[left_idx] > dists[right_idx]:
                    # left is smaller
                    left_idx, right_idx = right_idx, left_idx

                # add to keypoints_pair
                keypoints_pair['input_diameter'].append(dists[left_idx])
                keypoints_pair['output_diameter'].append(dists[right_idx])
                keypoints_pair['top_diameter'].append(dists[top_idx])
                keypoints_pair['length'].append(dist(kp_group[-2], kp_group[-1])) # revise later
                keypoints_pair['left_keypoints'].append(centers[left_idx])
                keypoints_pair['right_keypoints'].append(centers[right_idx])
                keypoints_pair['top_keypoints'].append(centers[top_idx])

            keypoints_pair_dict[dataset.imgs_files[batch[1][0]['image_id']]]={'image_info':keypoints_pair}
        # visualize
        # # annotate left and right keypoints in image using different color
        # for kp in keypoints_pair['left_keypoints']:
        #     print(tuple(kp))
        #     # round(kp[0]), round(kp[1])
        #     image = cv2.circle(image.copy(), (round(kp[0]), round(kp[1])), 1, (255,0,0), 1)

        # for kp in keypoints_pair['right_keypoints']:
        #     image = cv2.circle(image.copy(), (round(kp[0]), round(kp[1])), 1, (0,255,0), 1)

        # for kp in keypoints_pair['top_keypoints']:
        #     image = cv2.circle(image.copy(), (round(kp[0]), round(kp[1])), 1, (0,0,255), 1)

        # # save image
        # plt.figure(figsize=(20, 10))
        # plt.imshow(image)
        # plt.savefig("./Keypoint_Detection/nailfold_keypoint/model/show2.png")
        # plt.close()

        return keypoints_pair_dict
    
    keypoints_pair_dict = get_keypoint_gt_diameter(dataset_path)
    # print(keypoints_pair_dict)
 
    dataset_path = "./Keypoint_Detection/data/nailfold_dataset1/test"

    keypoints_pair_dict2 = get_keypoint_gt_diameter(dataset_path)
    # print(keypoints_pair_dict2)

    # merge two dict
    for key in keypoints_pair_dict2.keys():
        keypoints_pair_dict[key] = keypoints_pair_dict2[key]


    # save to result_dict_annotate.json
    with open("./Keypoint_Detection/nailfold_keypoint/model/results_dict_annotate.json", 'w') as f:
        json.dump(keypoints_pair_dict, f)
        


