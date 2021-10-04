import cv2
import sys
import os
import random
import numpy as np


class HorizontalFlip(object):
    def __init__(self, p=0.):
        self.p = p

    def __call__(self, img, labels, mode=None):
        aug_labels = labels.copy()
        if random.random() < self.p:
            img = np.fliplr(img)
            if mode == 'cxywha':    
                aug_labels[:, 1] = img.shape[1] - labels[:, 1]
                aug_labels[:, 5] = -labels[:, 5]
            if mode == 'xyxyxyxy':
                aug_labels[:, [0,2,4,6]] = img.shape[1] - labels[:, [0,2,4,6]]
            if mode == 'xywha':
                aug_labels[:, 0] = img.shape[1] - labels[:, 0]
                aug_labels[:, -1] = -labels[:, -1]                
        return img, aug_labels        


class VerticalFlip(object):
    def __init__(self ,p=0.):
        self.p = p

    def __call__(self, img, labels, mode=None):
        aug_labels = labels.copy()
        if random.random() < self.p:
            img = np.flipud(img)
            if mode == 'cxywha': 
                aug_labels[:, 2] = img.shape[0] - labels[:, 2]
                aug_labels[:, 5] = -labels[:, 5]
            if mode == 'xyxyxyxy':
                aug_labels[:, [1,3,5,7]] = img.shape[0] - labels[:, [1,3,5,7]]
            if mode == 'xywha':
                aug_labels[:, 1] = img.shape[0] - labels[:, 1]
                aug_labels[:, -1] = -labels[:, -1]   
        return img, aug_labels 

