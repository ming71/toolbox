import cv2
import sys
import os
import random
import numpy as np
import imgaug.augmenters as iaa

class Dropout(object):
    def __init__(self, p=0., **kwargs):
        self.p = p
        if not bool(kwargs):
            self.params = { 'p':(0.0, 0.15) }
        else:
            self.params = kwargs

    def __call__(self, img, labels, mode=None):
        aug_labels = labels.copy()
        if random.random() < self.p:
            aug = iaa.Dropout(**self.params)
            img = aug.augment_image(img)
                      
        return img, labels        



class CoarseDropout(object):
    def __init__(self, p=0., **kwargs):
        self.p = p
        if not bool(kwargs):
            self.params = { 'p':(0.0, 0.05),
                            'size_percent':(0.01, 0.05) }
        else:
            self.params = kwargs

    def __call__(self, img, labels, mode=None):
        aug_labels = labels.copy()
        if random.random() < self.p:
            aug = iaa.CoarseDropout(**self.params)
            img = aug.augment_image(img)
                      
        return img, labels      
