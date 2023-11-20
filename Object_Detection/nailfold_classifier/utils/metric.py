# coding:utf-8
import os
import cv2
import time
import numpy as np
import torch


def failure_analysis(X, y_hat, y_pred, epoch, stage, classes, save_dir):
    save_path = os.path.join(save_dir, "failure_examples", stage, str(epoch))
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    failure_index = y_pred.argmax(dim=1) != y_hat
    failure_data = X[failure_index]
    failure_label = y_pred.argmax(dim=1)[failure_index]
    true_label = y_hat[failure_index]

    failure_classes = []
    true_classes = []
    for label1, label2 in zip(failure_label.cpu().numpy(), true_label.cpu().numpy()):
        failure_classes.append(classes[label1])
        true_classes.append(classes[label2])

    for data, label1, label2 in zip(failure_data, true_classes, failure_classes):
        data = data.cpu().numpy().transpose(1, 2, 0)
        data = data * 255
        data = data.astype(np.uint8)
        timestamp = int(time.time() * 100000)
        cv2.imwrite(os.path.join(save_path, "{}-{}-{}.jpg".format(label1, label2, timestamp)), data[:, :, ::-1])


def evaluate_accuracy_and_loss(data_iter, model, loss, epoch, classes, save_dir, error_analysis=False, stage="val"):
    acc_sum = 0.0
    loss_sum = 0.0
    n = 0

    with torch.no_grad():
        for X, y in data_iter:
            X = X.cuda()
            y = y.cuda()
            y_pred = model(X)

            acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            loss_sum += loss(y_pred, y).sum().item()
            n += y.shape[0]

            if error_analysis:
                failure_analysis(X, y, y_pred, epoch, stage, classes, save_dir)
        return acc_sum / n, loss_sum / n
