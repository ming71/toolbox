import cv2
import sys
import os
import random
import numpy as np
import os.path as osp
from tqdm import tqdm

from dataset import *
from ops import *

if __name__ == "__main__":
    
    data_dir = 'data'
    dataset = FAIR1M(data_dir)

    pair = tqdm(zip(dataset.im_files, dataset.anno_files))
    for img_file, anno_file in pair:
        pair.set_description("Processing {}".format(img_file))
        filename = osp.splitext(osp.split(img_file)[1])[0]
        img = cv2.imread(img_file,1)
        classes, _bboxes = dataset.parse_annos(anno_file)
        transform = Augment([   
                                HSV(0.5, 0.5, p=0.5),
                                HorizontalFlip(p=0.8),
                                VerticalFlip(p=0.8),
                                Affine(degree=30, translate=0.2, scale=(0.8, 1.5), p=1),
                                # Noise(0.02, p=0.2),
                                Blur(1.3, p=0.5),
                                Cutout(p=0.5),
                                Dropout(p=0.5),
                                # CoarseDropout(p=0.5),
                                ], box_mode = 'xyxyxyxy')
        im, bboxes = transform(img, _bboxes.copy())
        dataset.save(im, classes, bboxes, filename)