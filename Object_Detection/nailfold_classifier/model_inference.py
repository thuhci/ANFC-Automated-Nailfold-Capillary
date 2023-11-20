# coding:utf-8
import os

import torch
import torchvision
from imutils.video import fps
from Object_Detection.nailfold_classifier.model_backbone import Backbone
from torch import nn
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from utils import transforms

classes = ["abnormal", "normal"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./checkpoints/resnet18/model-resnet18-2-0.8260869565217391.pth"

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

image_path = "./Object_Detection/data/mask_dataset/test/abnormal"
fps = fps.FPS()
fps.start()

with torch.no_grad():
    for image_name in os.listdir(image_path):
        file_path = os.path.join(image_path, image_name)
        img = read_image(file_path, mode=ImageReadMode.RGB)
        img = img.to(device)
        img = img.unsqueeze(dim=0)
        img = val_transforms(img).to(device)

        res = model(img)
        cls_index = res.argmax(dim=1)
        cls_prob = nn.functional.softmax(res, dim=1)

        pred_prob = cls_prob[0][cls_index].item()
        pred_cls = classes[cls_index]

        print(pred_prob, pred_cls)

        fps.update()
fps.stop()
print("FPS: {}".format(fps.fps()))
print("time: {}".format(fps.elapsed()))
