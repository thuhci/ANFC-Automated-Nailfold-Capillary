import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from nailfold_data_preprocess.utils.my_parser import get_parse

def main(args):
    dataset_dir = args.input
    if args.input is None:
        dataset_dir = "../data/classify_dataset"
    dataset = os.path.join(dataset_dir, 'train')
    class_list = os.listdir(dataset) 
    num_classes = dict()
    for classes in class_list:
        dataset = os.path.join(dataset_dir, 'train', classes)
        num_classes[classes] = len(os.listdir(dataset))

    print(num_classes)
    x = [ num_classes[key] for key in num_classes]
    labels = [ key for key in num_classes]
    plt.pie(np.array(x),autopct="(%1.1f%%)",labels=labels)
    plt.savefig("a.png")

if __name__ == "__main__":
    args= get_parse()
    main(args)
    