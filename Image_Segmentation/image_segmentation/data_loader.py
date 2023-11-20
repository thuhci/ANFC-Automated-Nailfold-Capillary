import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train', augmentation_prob=0.4, img_ch=1):
        """Initializes image paths and preprocessing module."""
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.root = root
        self.image_size = image_size
        self.augmentation_prob = augmentation_prob
        self.img_ch = img_ch
        self.rotation_degree_bin = [0, 10, 20, 30] # revised by zhaolx

        self.GT_paths = os.path.join(root, 'masks/')
        self.image_paths = list(map(lambda x: os.path.join(os.path.join(
            root, 'images/'), x), os.listdir(os.path.join(root, 'images/'))))

        print("image count in {} path :{}".format(
            self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        # load image and GT
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1][:-len(".jpg")]
        # print("test:",self.GT_paths,filename)
        GT_path = self.GT_paths + filename + '_mask.png'

        image = Image.open(image_path)
        GT = Image.open(GT_path)

        # augmentation
        # both: crop,     resize to self.image_size,    norm
        # train: random size crop(important), rotation, flip, color jitter

        # present code:
        # both: crop,     resize to self.image_size,     norm
        # train: random_resize, centre_crop, rotation, flip, color jitter

        
        # image transformation
        
        Transform_ = self.image_transform(image.size[1], image.size[0], GT)
        image = Transform_(image)
        GT = Transform_(GT)

        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        if len(image.shape) == 4:
            if self.img_ch == 1:
                image = image[:,1:2,:,:] # only use green channel
            if GT.shape[1] > 1:
                GT = GT[:,0:1,:,:] # ncrops, c, h, w
        else:
            if self.img_ch == 1:
                image = image[1:2,:,:]
            if GT.shape[0] > 1:
                GT = GT[0:1,:,:]
        image = image.view(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        GT = GT.view(-1, GT.shape[-3], GT.shape[-2], GT.shape[-1])
        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)
    

    def image_transform(self, h, w, GT):
        aspect_ratio = h / w
        seed = np.random.randint(100)
        random.seed(seed)

        Transform_size = []

        if (self.mode == 'train'):

            if random.random() <= self.augmentation_prob:
                rotation_degree = self.rotation_degree_bin[random.randint(0, 3)] + random.randint(0, 10)
                Transform_size.append(T.RandomRotation(
                    (rotation_degree, rotation_degree)))
                
                Transform_size.append(T.RandomVerticalFlip(random.randint(0,10)==0))
                Transform_size.append(T.RandomHorizontalFlip(random.randint(0,2)%2==0))
            
            if random.random() <= self.augmentation_prob:
                Transform_size.append(T.ColorJitter(brightness=0.1, contrast=0.05, hue=0.2))

            gt_sum = np.sum(np.sum(np.array(GT)[:,:,0]))/255            
            if gt_sum > 30000: # 3, 768, 1024 original size
                resize_w = random.randint(round(max(self.image_size,0.5*w)),round(0.9*w))
                Transform_size.append(
                    T.FiveCrop((int(resize_w*aspect_ratio), resize_w)))
                Transform_size.append(T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])))
            else:
                Transform_size.append(T.ToTensor())

            Transform_size.append(
                T.Resize((int(self.image_size*aspect_ratio)-int(self.image_size*aspect_ratio) % 16, self.image_size)))
        

        elif (self.mode == 'valid' or self.mode == 'test'):
            Transform_size.append(
                T.Resize((int(self.image_size*aspect_ratio)-int(self.image_size*aspect_ratio) % 16, self.image_size)))
            Transform_size.append(T.ToTensor())
        
        Transform_size = T.Compose(Transform_size)

        return Transform_size


def collate_fn(batch):
    for i, sample in enumerate(batch):
        if i == 0:
            X = sample[0]
            y = sample[1]
        else:
            X = torch.cat((X,sample[0]),0)
            y = torch.cat((y,sample[1]),0)
    return X,y


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4, img_ch=1):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size,
                          mode=mode, augmentation_prob=augmentation_prob, img_ch=img_ch)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader



if __name__ == "__main__":
    import cv2
    output_dir = './Image_Segmentation/image_segmentation/output'
    train_loader = get_loader(image_path='./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/train', image_size=256, batch_size=4, mode='train', augmentation_prob=0.4, img_ch=1)
    for i, (image, GT) in enumerate(train_loader):
        image = image.view(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        GT = GT.view(-1, GT.shape[-3], GT.shape[-2], GT.shape[-1])
        print(image.shape)
        # save image
        ncrops, c, h, w = image.shape
        for n in range(ncrops):
            # concate n images and save
            img = ((1+image[n])/2).permute(1,2,0).numpy()
            gt = GT[n].permute(1,2,0).numpy()
            # print(np.max(img), np.min(img), np.max(gt), np.min(gt))
            whole_image = np.concatenate((img,gt), axis=0)
            cv2.imwrite(f'{output_dir}/train_loader_{i}_{n}.jpg', whole_image*255)
        if i == 5:
            break