# coding:utf-8
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import numpy as np
import random
import torchvision
from torch.utils.data import Dataset, DataLoader


class NpyDataset(Dataset):
    def __init__(self, filenames, labels, transforms, down_sample=False):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.down_sample = down_sample

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(self.filenames[idx]).astype(np.float32)
        if self.down_sample:
            data = data[:, ::4]
        data = self.transforms(data)
        # print(data.shape)
        return data, self.labels[idx]


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, transforms):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = read_image(self.filenames[idx], mode=ImageReadMode.RGB)
        image = self.transforms(image)
        return image, self.labels[idx]


def fetch_dataloader(dataset_dir, ratio, train_transforms, val_transforms, trainset_size,
                     batchsize=512, num_workers=8, seed=100):
    random.seed(seed)

    dataset = torchvision.datasets.ImageFolder(dataset_dir)
    classes = dataset.classes
    character = [[] for _ in range(len(classes))]
    random.shuffle(dataset.samples)
    sample_count = {}

    for x, y in dataset.samples:
        if y not in sample_count:
            sample_count[y] = 0
        if sample_count.get(y, 0) >= trainset_size.get(classes[y]):
            continue
        character[y].append(x)
        sample_count[y] += 1

    for i, x, in enumerate(character):
        print("{} : {}".format(classes[i], len(x)))

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
        for x in data[num_val_index:]:
            test_inputs.append(str(x))
            test_labels.append(i)

    print("train_inputs: {}, train_labels: {}".format(len(train_inputs), len(train_labels)))
    print("val_inputs: {}, val_labels: {}".format(len(val_inputs), len(val_labels)))
    print("test_inputs: {}, test_labels: {}".format(len(test_inputs), len(test_labels)))

    train_dataset = ImageDataset(train_inputs, train_labels, train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, drop_last=True,
                                  shuffle=True, num_workers=num_workers)

    val_dataset = ImageDataset(val_inputs, val_labels, val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, drop_last=False,
                                shuffle=False, num_workers=num_workers)

    test_dataset = ImageDataset(test_inputs, test_labels, val_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, drop_last=False,
                                 shuffle=False, num_workers=num_workers)

    loader = {}
    loader["train_loader"] = train_dataloader
    loader["val_loader"] = val_dataloader
    loader["test_loader"] = test_dataloader
    return loader, classes
