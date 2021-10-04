
import random


class Augment(object):
    def __init__(self, augmentations, probs=1, box_mode=None):
        self.augmentations = augmentations
        self.probs = probs
        self.mode = box_mode
        
    def __call__(self, img, labels):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                img, labels = augmentation(img, labels, self.mode)

        return img, labels

