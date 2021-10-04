import cv2
import sys
import os
import random
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

'''
def __init__(self,
                 nb_iterations=2,
                 position="uniform",
                 size=0.02,
                 squared=False,
                 fill_mode="uniform",
                 cval=128,
                 fill_per_channel=0.5,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
'''

class Cutout(object):
    def __init__(self, p=0.5, **kwargs):
        self.p = p
        if not bool(kwargs):
            self.params = {  'nb_iterations':20,
                        'position':"uniform",
                        'size':0.02,
                        'squared':False,
                        'fill_mode':"gaussian",
                        'fill_per_channel':0.5}
        else:
            self.params = kwargs

    def __call__(self, img, labels, mode=None):
        aug_labels = labels.copy()
        if random.random() < self.p:
            aug = iaa.Cutout(**self.params)
            img = aug.augment_image(img)
                      
        return img, labels        


