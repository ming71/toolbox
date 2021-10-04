import cv2
import sys
import os
import random
import numpy as np
import imgaug.augmenters as iaa


class HSV(object):
    def __init__(self , saturation=0, brightness=0, p=0.):
        self.saturation = saturation 
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(-1, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels
    

class Blur(object):
    def __init__(self, sigma=0 ,p=0.):
        self.sigma = sigma 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0,self.sigma))
            img = blur_aug.augment_image(img)
        return img, labels


class Grayscale(object):
    def __init__(self, grayscale=0. ,p=0.):
        self.alpha = random.uniform(grayscale,1.0)
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, labels


class Gamma(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            gm = random.uniform(1-self.intensity,1+self.intensity)
            img = np.uint8(np.power(img/float(np.max(img)), gm)*np.max(img))
        return img, labels


class Noise(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, labels



class Sharpen(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity,1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, labels


class Contrast(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            contrast_aug = aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img=contrast_aug.augment_image(img)
        return img, labels


