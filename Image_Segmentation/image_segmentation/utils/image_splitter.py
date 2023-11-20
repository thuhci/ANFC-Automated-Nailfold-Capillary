
import numpy as np
import torch

class ImageSplitter:
    def __init__(self, img_h, img_w, img_c, num_patches=2):
        '''
        img_h: int, image height
        img_w: int, image width
        img_c: int, image chanels
        num_patches: int, split into num_patches*num_patches patches
        '''
        self.patches = []
        self.h = img_h
        self.w = img_w
        self.chanels = img_c
        self.num_patches = num_patches

    def split_into_patch(self, image: np.ndarray):
        '''
        args
            image: np.ndarray, (height, width, chanels)
        return
            patches: list[np.ndarray], [(patch_height, patch_width, chanels)]
        '''
        if self.num_patches == 1:
            return [image]
        height, width, c = image.shape
        assert height == self.h and width == self.w and self.chanels == c

        patch_height = height // self.num_patches
        patch_width = width // self.num_patches

        self.patches = []
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                patch = image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                self.patches.append(patch)

        return self.patches

    def reverse(self, masks: torch.Tensor):
        '''
        args
            masks: torch.Tensor, (batch_size, 1, resize_patch_height, resize_patch_width)
        return
            images: np.ndarray, (batch_size, height, width)
        '''
        if self.num_patches == 1:
            return masks
        
        batch_size, chanel, patch_height, patch_width = masks.shape
        height = patch_height * self.num_patches
        width = patch_width * self.num_patches
        batch_size = batch_size // (self.num_patches**2)

        images = np.zeros((batch_size, chanel, height, width))
        for b in range(batch_size):
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    images[b, :, i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = masks[b * self.num_patches* self.num_patches + i * self.num_patches + j]

        return images
