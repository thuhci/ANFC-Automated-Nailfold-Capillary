import cv2
import os
import json
import numpy as np

H = 768
W = 1024

def main():
    raw_dir = './data/tangshan_segmentation'
    dataset_dir = './data/segment_dataset'
    output_folder = os.path.join(dataset_dir, 'train/masks')
    image_folder = os.path.join(dataset_dir, 'train/images')
    test_output_folder = os.path.join(dataset_dir, 'test/masks')
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
        new_json_path = os.path.join(output_dir, file.replace('.jpg', '_mask.png'))

        with open(lab_path,"r") as f:    
            anno = json.load(f)
        img = cv2.imread(img_path)
        plys = [shape['points'] for shape in anno['shapes']]
        mask = np.zeros((img.shape), dtype=np.uint8)
        for ply in plys:
            ply = np.array(ply).astype(np.int32)
            mask = cv2.fillPoly(mask, [ply], color=(255, 255, 255))
        cv2.imwrite(new_json_path, mask)
        os.system(f'cp {img_path} {image_dir}/{file}')
        # os.system(f'mv {json_path} {new_json_path}')
        print(cnt)


main()