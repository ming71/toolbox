import cv2
import sys
import os
import random
import numpy as np


def random_affine(img,  targets=(), degree=10, translate=.1, scale=.1, shear=10):
    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    # # # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    if not isinstance(scale, tuple):
        s = random.uniform(1 - scale, 1 + scale)
    else:
        s = random.uniform(scale[0], scale[1])
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)


    M =  T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))  # BGR order borderValue

    # Return warped points also
    t = targets.copy()
    targets[:, [0,2,4,6]] = t[:, [0,2,4,6]] * M[0,0] + t[:, [1,3,5,7]] * M[0,1] + M[0,2]
    targets[:, [1,3,5,7]] = t[:, [0,2,4,6]] * M[1,0] + t[:, [1,3,5,7]] * M[1,1] + M[1,2]
  
    return imw, targets

    
class Affine(object):
    def __init__(self, degree = 0., translate = 0., scale = 0., shear = 0., p=0.):
        self.degree = degree 
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            if mode == 'xywha':
                labels = rbox_2_quad(labels, mode = 'xywha')
                img, labels = random_affine(img, labels, 
                            degree=self.degree,translate=self.translate,
                            scale=self.scale,shear=self.shear ) 
                labels = quad_2_rbox(labels, mode = 'xywha')

            else:
                img, labels = random_affine(img, labels, 
                                degree=self.degree,translate=self.translate,
                                scale=self.scale,shear=self.shear ) 
        return img, labels 


