import os

import cv2
import numpy as np
import torch


class ImageVisualization():
    # visualize image, predicted segementation, ground truth segementation and save them
    def __init__(self, image_size, save_path='Image_Segmentation/image_segmentation/result/visualization'):
        self.save_path = save_path
        self.w = image_size

    def check_path(self, img_name):
        # check if the path exists
        if not os.path.exists(os.path.join(self.save_path,img_name)):
            os.makedirs(os.path.join(self.save_path,img_name))
        
    def check_image(self, image):
        # if the image is a tensor, convert it to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        # if the image is a grayscale image, convert it to a 3-channel image
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
            image = np.concatenate([image, image, image], axis=2)
        # if the image shape is (H,W,C), convert it to (C,H,W)
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = image.transpose((1,2,0))
        # from grey to rgb image
        if image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)
        # if the image is 0-1, convert it to 0-255
        if image.max() <= 1 and image.min() >= 0:
            image = image * 255
        # if the image is -1to1, convert it to 0-255
        if image.min() < 0 and image.min() >= -1:
            image = (image + 1) * 127.5
        # if resize to self.w
        self.h = int(image.shape[0]/image.shape[1]*self.w)
        if image.shape[0] != self.h or image.shape[1] != self.w:
            image = cv2.resize(image, (self.w, self.h))

        return image.astype(np.uint8)


    def visualize(self, image, img_name):
        # visualize and save
        image = self.check_image(image)

        cv2.imwrite(os.path.join(self.save_path,f'{img_name}.png'), image)
        return image

    def visualize_pred_gt(self, img_name, image, pred_segmentation, gt_segmentation, epoch=0):
        # compare predicted segmentation and ground truth segmentation
        image = self.check_image(image)
        pred_segmentation = self.check_image(pred_segmentation)
        gt_segmentation = self.check_image(gt_segmentation)

        merge_image = np.zeros((image.shape[0]*2, image.shape[1], image.shape[2]))
        merge_image[:image.shape[0], :, :] = image
        merge_image[image.shape[0]:, :, 0] = pred_segmentation[:,:,0]
        merge_image[image.shape[0]:, :, 1] = gt_segmentation[:,:,0]

        self.check_path("visualize_pred_gt")
        cv2.imwrite(os.path.join(self.save_path,"visualize_pred_gt",f'{img_name}_{epoch + 1}.png'), merge_image)
        return merge_image

    def visualize_mask(self, image, mask, img_name):
        # visualize image and mask
        image = self.check_image(image)
        mask = self.check_image(mask)

        merge_image = np.zeros((image.shape[0]*2, image.shape[1], image.shape[2]))
        merge_image[:image.shape[0], :, :] = image
        merge_image[image.shape[0]:, :, :] = mask

        self.check_path("visualize_mask")
        cv2.imwrite(os.path.join(self.save_path,'visualize_mask',f'{img_name}.png'), merge_image)
        return merge_image


if __name__ == '__main__':
    from Image_Segmentation.image_segmentation.image2segment import \
        t_images2masks

    data_path = "../Nailfold_Data_Tangshan/tangshan_data/tangshan_segmentation"
    names = ["8_49510_1.jpg","8_56826_2.jpg", "9_59673_2.jpg", "9_59673_1.jpg", "9_59673_4.jpg", "9_62506_4.jpg", "8_56826_1.jpg"]
    imgs = []
    for name in names:
        train_dir = os.path.join(data_path, name)
        img = cv2.imread(train_dir)
        imgs.append(img)
        print(name)
    masks = t_images2masks(imgs, 1)
    print(masks[0].shape)
    
    vis = ImageVisualization(512)
    vis.visualize_pred_gt(names[0], imgs[0], masks[0], masks[0])
    vis.visualize_pred_gt(names[1], imgs[1], masks[1], masks[0])
    # for i in range(len(imgs)):
    #     vis.visualize_mask(imgs[i], masks[i], names[i])