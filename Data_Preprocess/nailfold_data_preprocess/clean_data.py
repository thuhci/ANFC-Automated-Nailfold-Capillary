from Video_Process.nailfold_video_process.blur_detection import estimate_blur
import os
import cv2
from nailfold_data_preprocess.utils.my_parser import get_parse
import numpy as np
import matplotlib.pyplot as plt


def get_blur(image: np.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    # blur_map[np.abs(blur_map) < 5] = 0
    score = np.max(blur_map)
    return blur_map, score, bool(score < threshold)

def get_images(args):
    dataset_dir = args.input
    if args.input is None:
        dataset_dir = "../data/classify_dataset"
    dataset = os.path.join(dataset_dir, 'train')
    class_list = ['hemo'] # os.listdir(dataset) 
    num_classes = dict()
    scores = []
    for classes in class_list:
        dataset = os.path.join(dataset_dir, 'train', classes)
        num_classes[classes] = len(os.listdir(dataset))
        files = []
        for file in os.listdir(dataset):
            img_path = os.path.join(dataset, file)
            img = cv2.imread(img_path)
            blur_map, score, blurry = get_blur(img)
            print(file,score)
            scores.append(score)
            if score > 30:
                files.append(img_path)
                
    plt.plot(np.arange(len(scores)),np.array(scores))
    plt.savefig("b.png")

    os.makedirs("data", exist_ok=True)
    for file in files:
        os.system(f"cp {file} data")




if __name__ == "__main__":
    args= get_parse()
    get_images(args)
    