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
from torch.utils.data import DataLoader
from utils.metric import evaluate_accuracy_and_loss, failure_analysis
from utils.transforms import ZeroOneNormalize

warnings.filterwarnings("ignore")

os.environ["TORCH_HOME"] = "./pretrained_models"

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./data/classifier_cifar100.yaml", type=str, help="data file path")
args = parser.parse_args()
cfg_path = args.cfg

with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
print(cfg_dict)

visible_device = cfg_dict.get("device")
batchsize = cfg_dict.get("batch_size")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(256, 256)).to(device),
    torchvision.transforms.RandomCrop(size=(224, 224)),
    torchvision.transforms.ColorJitter(brightness=brightness, hue=hue, saturation=saturation, contrast=contrast),
    torchvision.transforms.RandomHorizontalFlip(p=left_right_flip),
    torchvision.transforms.RandomVerticalFlip(p=up_down_flip),
    torchvision.transforms.RandomRotation(degrees=rotate_degree),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(224, 224)).to(device),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)

cifar100_train = torchvision.datasets.CIFAR100(root="./data", train=True, transform=train_transforms, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root="./data", train=False, transform=val_transforms, download=True)

train_data_loader = DataLoader(cifar100_train, batch_size=batchsize, drop_last=True, shuffle=True,
                               num_workers=num_workers)
test_data_loader = DataLoader(cifar100_test, batch_size=batchsize, drop_last=False, shuffle=False,
                              num_workers=num_workers)
classes = cifar100_train.classes
print("train: {}, test: {}, classes: {}".format(len(train_data_loader), len(test_data_loader), len(classes)))


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

            if batch_idx % 20 == 0:
                print("epoch: {}, iter: {}, lter loss: {:.4f}, iter acc: {:.4f}".format(epoch, batch_idx, l.item(), (
                        y_pred.argmax(dim=1) == y).float().mean().item()))
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
            best_model = os.path.join(os.path.join(save_dir, model_name),
                                      "model-{}-{}-{}.pth".format(model_name, epoch, best_acc))
            torch.save(model.module.state_dict(), best_model)
    return train_acc, train_loss, val_acc, val_loss, lr_decay_list, best_model, best_acc


for modle_name in [
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
    "resnet18", "resnet34", "resnet50", "resnet101",
    "resnext50_32x4d", "resnext101_32x8d",
    "densenet121", "densenet161", "densenet169",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
    "squeezenet1_0", "squeezenet1_1",
    "wide_resnet50_2", "wide_resnet101_2"
]:
    print("=" * 40, "train {}".format(model_name), "=" * 40)
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
    print(model)
    print(model.device_ids)

    train_iter, val_iter, test_iter = train_data_loader, val_data_loader, test_data_loader
    loss = torch.nn.CrossEntropyLoss()

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
    plt.suptitle(model_name)
    plt.savefig(os.path.join(save_dir, model_name, "{}.jpg".format(model_name)))
    plt.show()
