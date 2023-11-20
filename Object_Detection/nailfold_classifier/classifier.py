# coding:utf-8
import os

import cv2
import numpy as np
import torch
import torchvision
from Object_Detection.nailfold_classifier.model_backbone import Backbone
from Object_Detection.nailfold_classifier.utils import transforms
from torch import nn
from torchvision.transforms import ToTensor


def t_classify_abnormal(imgs: list[np.array], model_path="checkpoints/resnet18/abnormal-normal-clean-80-50.pth"):
    """
    Classify the category of blood vessel images.

    Args:
        imgs (list of np.array): List of blood vessel images in BGR format.
        model_path (str, optional): Path to the classification model checkpoint. Defaults to "checkpoints/resnet18/abnormal-normal-clean-80-50.pth".

    Returns:
        tuple: A tuple containing the following elements:
            - scores (list): List of classification scores.
            - classes (list): List of class labels, where "abnormal" corresponds to class 0 and "normal" to class 1.
    """
    classes = ["abnormal", "normal"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cur_path=os.path.abspath(__file__)
    cur_path = os.path.dirname(cur_path)
    model_path = os.path.join(cur_path,model_path)

    val_transforms_list = [
        torchvision.transforms.Resize(size=(224, 224)),
        transforms.ZeroOneNormalize(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_transforms = torchvision.transforms.Compose(val_transforms_list)

    backbone = Backbone(out_dimension=len(classes), model_name="resnet18")
    model, _, _ = backbone.build_model()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.to(device)
    model.eval()

    with torch.no_grad():
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        imgs = [(ToTensor()(img)*255).to(device) for img in imgs]
        
        # img = img.to(device)
        # img = img.unsqueeze(dim=0)
        imgs = [val_transforms(img).to(device) for img in imgs]
        img = torch.stack(imgs)
        res = model(img)
        cls_index = res.argmax(dim=1).cpu().numpy().tolist()
        cls_prob = nn.functional.softmax(res, dim=1).cpu().numpy()

        pred_prob = cls_prob[...,cls_index][:,0].tolist()
        pred_cls = [classes[i] for i in cls_index]

        return (pred_prob, pred_cls)


if __name__ == "__main__":
    dataset = "dataset/train/abnormal"
    imgs = []
    file_names = os.listdir(dataset)
    for file in file_names:
        img = cv2.imread(os.path.join(dataset, file))
        imgs.append(img)
    scores, classes = t_classify_abnormal(imgs)
    output = "results/visualize"
    os.makedirs(output, exist_ok=True)
    for name, img, score, label in zip(file_names, imgs, scores, classes):
        cv2.putText(img, label, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output,name), img)