# coding:utf-8
import argparse
import os
import shutil
import warnings

import torch
import torchvision
import yaml
from matplotlib import pyplot as plt
from Object_Detection.nailfold_classifier.model_backbone import Backbone
from utils import transforms
from utils.data_loader import fetch_dataloader
from utils.metric import evaluate_accuracy_and_loss, failure_analysis

warnings.filterwarnings("ignore")

os.environ["TORCH_HOME"] = "./pretrained_models"

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./data/classifier.yaml", type=str, help="data file path")
args = parser.parse_args()
cfg_path = args.cfg

with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
print(cfg_dict)

dataset_dir = cfg_dict.get("dataset_dir")
trainset_size = cfg_dict.get("train_size")
visible_device = cfg_dict.get("device")
batchsize = cfg_dict.get("batch_size")
train_ratio = cfg_dict.get("train_ratio")
val_ratio = cfg_dict.get("val_ratio")
test_ratio = cfg_dict.get("test_ratio")
# data augmentation
invert_ratio = cfg_dict.get("invert_ratio", 0.0)
brightness = cfg_dict.get("brightness", 0.0)
hue = cfg_dict.get("hue", 0.0)
saturation = cfg_dict.get("saturation", 0.0)
contrast = cfg_dict.get("contrast", 0.0)
left_right_flip = cfg_dict.get("left_right_flip", 0.0)
up_down_flip = cfg_dict.get("up_down_flip", 0.0)
rotate_degree = cfg_dict.get("rotate_degree", 0)
img_norm = cfg_dict.get("img_norm", False)

num_workers = cfg_dict.get("num_workers")
num_epoches = cfg_dict.get("epoch")
model_name = cfg_dict.get("model_name")
lr = cfg_dict.get("lr")
step_size = cfg_dict.get("step_size")
gamma = cfg_dict.get("gamma")
weight_decay = cfg_dict.get("weight_decay")
save_dir = cfg_dict.get("save_dir")

train_transforms_list = [
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ColorJitter(brightness=brightness, hue=hue, saturation=saturation, contrast=contrast),
    torchvision.transforms.RandomHorizontalFlip(p=left_right_flip),
    torchvision.transforms.RandomVerticalFlip(p=up_down_flip),
    torchvision.transforms.RandomRotation(degrees=rotate_degree),
    transforms.ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms_list = [
    torchvision.transforms.Resize(size=(224, 224)),
    transforms.ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)

data_loader, classes = fetch_dataloader(dataset_dir, ratio=[train_ratio, val_ratio, test_ratio],
                                        batchsize=batchsize, num_workers=num_workers, trainset_size=trainset_size,
                                        train_transforms=train_transforms, val_transforms=val_transforms)
train_data_loader = data_loader["train_loader"]
val_data_loader = data_loader["val_loader"]
test_data_loader = data_loader["test_loader"]
print(classes)
print("train: {}, val: {}, test: {}".format(len(train_data_loader), len(val_data_loader), len(test_data_loader)))

backbone = Backbone(out_dimension=len(classes), model_name=model_name, pretrained=True)
model, train_params, pretrained_params = backbone.build_model()
optimizer = torch.optim.SGD(
    [
        {"params": train_params, "lr": lr},
        {"params": pretrained_params, "lr": lr / 10}
    ],
    lr=lr,
    weight_decay=weight_decay
)
model = torch.nn.DataParallel(model, device_ids=visible_device).cuda()
# model = model.cuda() #to(device="cuda:1")
print(model)
print(model.device_ids)

train_iter, val_iter, test_iter = train_data_loader, val_data_loader, test_data_loader
loss = torch.nn.CrossEntropyLoss()


def train(model, train_iter, val_iter, loss, num_epoches, optimizer):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    lr_decay_list = []

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_acc = 0.0
    best_model = ""

    for epoch in range(num_epoches):
        lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        model.train()

        for batch_idx, (X, y) in enumerate(train_iter):
            X = X.cuda()
            y = y.cuda()
            y_pred = model(X)

            l = loss(y_pred, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

            # if batch_idx % 20 == 0:
            #     print("epoch: {}, iter: {}, lter loss: {:.4f}, iter acc: {:.4f}".format(epoch, batch_idx, l.item(), (
            #             y_pred.argmax(dim=1) == y).float().mean().item()))
        lr_scheduler.step()
        model.eval()

        # t_acc, t_loss = evaluate_accuracy_and_loss(train_iter, model, loss, epoch, classes, save_dir,
        #                                            error_analysis=True, stage="train")
        v_acc, v_loss = evaluate_accuracy_and_loss(val_iter, model, loss, epoch, classes, save_dir,
                                                   error_analysis=True, stage="val")
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_loss_sum / n)
        # train_acc.append(t_acc)
        # train_loss.append(t_loss)
        val_acc.append(v_acc)
        val_loss.append(v_loss)

        print("epoch: {}, train acc: {:.4f}, train loss: {:.4f}, val acc: {:.4f}, val loss: {:.4f}".format(
            epoch, train_acc[-1], train_loss[-1], val_acc[-1], val_loss[-1]))
        if v_acc > best_acc:
            if os.path.exists(os.path.join(save_dir, model_name)) is False:
                os.makedirs(os.path.join(save_dir, model_name))
            best_acc = v_acc
            best_model = os.path.join(os.path.join(save_dir, model_name), "model-{}-{}-{}.pth".format(model_name, epoch, best_acc))
            torch.save(model.module.state_dict(), best_model)
    return train_acc, train_loss, val_acc, val_loss, lr_decay_list, best_model, best_acc


if os.path.exists(os.path.join(save_dir, "failure_examples")):
    shutil.rmtree(os.path.join(save_dir, "failure_examples"))
train_acc, train_loss, val_acc, val_loss, lr_decay_list, best_model, best_acc = train(
    model, train_iter, val_iter, loss, num_epoches, optimizer)
print("best model: {}, best accuracy: {}".format(best_model, best_acc))

model, _, _ = backbone.build_model()
model.load_state_dict(torch.load(best_model))
model = model.cuda()
model.eval()
test_acc, test_loss = evaluate_accuracy_and_loss(test_data_loader, model, loss, "test", classes, save_dir,
                                                 error_analysis=True, stage="val")
print("test accuracy: {}, test loss: {}".format(test_acc, test_loss))

fig, axes = plt.subplots(1, 3)
axes[0].plot(list(range(1, num_epoches + 1)), train_loss, color="r", label="train loss")
axes[0].plot(list(range(1, num_epoches + 1)), val_loss, color="b", label="validate loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(list(range(1, num_epoches + 1)), train_acc, color="r", label="train acc")
axes[1].plot(list(range(1, num_epoches + 1)), val_acc, color="b", label="validate acc")
axes[1].legend()
axes[1].set_title("Accuracy")

axes[2].plot(list(range(1, num_epoches + 1)), lr_decay_list, color="r", label="lr")
axes[2].legend()
axes[2].set_title("Learning Rate")
plt.savefig(os.path.join(save_dir, "loss.png"))
