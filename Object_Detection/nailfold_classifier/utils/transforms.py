# coding:utf-8
import numpy as np
from PIL import ImageOps as plops
import random
from PIL import Image, ImageFilter


class InvertTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        random_prob = np.random.uniform(0, 1)
        if 1.0 - random_prob < self.p:
            img = plops.invert(img)
        return img


class UpDownFlipTransform(object):
    """
        Image图片数据随机上下翻转
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        random_prob = np.random.uniform(0, 1)
        if 1.0 - random_prob < self.p:
            img = img[::-1, :].copy()
        return img


class LeftRightFlipTransform(object):
    """
        Image图片数据随机左右翻转
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        random_prob = np.random.uniform(0, 1)
        if 1.0 - random_prob < self.p:
            img = img[:, ::-1].copy()
        return img


class MinMaxNormalize(object):
    """
        最大-最小归值一化

    """

    def __call__(self, x):
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)
        return x


class MeanStdNormalize(object):
    def __call__(self, x):
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / x_std
        return x


class ZeroMeanNormalize(object):
    def __call__(self, x):
        x_mean = x.mean()
        x = x - x_mean
        return x


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class AddSaltPepperNoise(object):
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class AddBlur(object):
    def __init__(self, p=0.5, blur="normal"):
        self.p = p
        self.blur = blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if self.blur == "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            if self.blur == "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img
            if self.blur == "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img
        return img
