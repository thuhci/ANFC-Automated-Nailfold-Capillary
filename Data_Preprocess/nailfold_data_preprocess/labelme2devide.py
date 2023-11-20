import cv2
import os
import json
import numpy as np

H = 768
W = 1024

def main():
    raw_dir = './data/tangshan_segmentation_clean'
    dataset_dir = './data/instance_segment_dataset'
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
        
        if cnt > len(files) * 3 / 4:
            output_dir = test_output_folder
            image_dir = test_image_folder
        else:
            output_dir = output_folder
            image_dir = image_folder
        os.makedirs(output_dir,exist_ok=True)
        os.makedirs(image_dir,exist_ok=True)
        
        new_json_path = os.path.join(output_dir, file.replace('.png', '.json').replace('.jpg', '.json'))
        os.system(f'cp {img_path} {image_dir}/{file}')
        os.system(f'cp {lab_path} {new_json_path}')
        # os.system(f'mv {json_path} {new_json_path}')
        print(output_dir,cnt)


main()