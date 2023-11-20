# coding:utf-8
import torch


def get_dataset_mean_and_std(dataset, batch_size):
    """
    计算数据集的均值和方差，用来做归一化
    :param dataset:
    :param batch_size:
    :return:
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    total_count = 0

    for inputs, targets in dataloader:
        # 音频特征数据是1通道的，所以这里是1
        for i in range(1):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
            total_count += inputs.shape
    mean.div_(total_count)
    std.div_(total_count)
    return mean, std
