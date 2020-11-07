import cv2
import sys
import os
import math
import glob
import torch
import random
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from shapely.geometry import Polygon

##
def makedir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# 统计每个文件中物体类别数目
# CLASSES是类别名
def category_statistics(labels, CLASSES=None):
    res_cnt = {}
    box_areas = []
    for classname in CLASSES:
        res_cnt[classname] = 0
    pbar = tqdm(labels)
    for label in pbar:
        pbar.set_description("category statistics")
        with open(label, 'r') as f:
            objs = f.readlines()
            for obj in objs:
                classname, *bbox = obj.strip().split()
                box_areas.append(bbox_area(bbox))
                assert classname in CLASSES, 'wrong classname in '.format(filename)
                res_cnt[classname] += 1
    return res_cnt, box_areas


def augment_ratio(cnt):
    objects = [x for x in  cnt.values()]
    rates = [int(max(objects) / x) - 1 for x in objects]
    rates[0] = int(rates[0]/10)  # [117, 857, 756, 904, 574]
    scheduler = cnt.copy()
    for idx, classname in enumerate(scheduler.keys()):
        scheduler[classname] = rates[idx]
    print(scheduler)
    return scheduler


def bbox_area(bbox):
    bbox = np.asarray(bbox)
    bbox = np.array(bbox).reshape(4, 2)
    poly = Polygon(bbox).convex_hull
    return poly.area


## bbox trans 
def quad_2_rbox(quads, mode='xyxya'):
    # http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg
    if len(quads.shape) == 1:
        quads = quads[np.newaxis, :]
    rboxes = np.zeros((quads.shape[0], 5), dtype=np.float32)
    for i, quad in enumerate(quads):
        rbox = cv2.minAreaRect(quad.reshape([4, 2]))    
        x, y, w, h, t = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]
        if np.abs(t) < 45.0:
            rboxes[i, :] = np.array([x, y, w, h, t])
        elif np.abs(t) > 45.0:
            rboxes[i, :] = np.array([x, y, h, w, 90.0 + t])
        else:   
            if w > h:
                rboxes[i, :] = np.array([x, y, w, h, -45.0])
            else:
                rboxes[i, :] = np.array([x, y, h, w, 45])
    # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    if mode == 'xyxya':
        rboxes[:, 0:2] = rboxes[:, 0:2] - rboxes[:, 2:4] * 0.5
        rboxes[:, 2:4] = rboxes[:, 0:2] + rboxes[:, 2:4]
    rboxes[:, 0:4] = rboxes[:, 0:4].astype(np.int32)
    return rboxes

def rbox_2_quad(rboxes, mode='xyxya'):
    if len(rboxes.shape) == 1:
        rboxes = rboxes[np.newaxis, :]
    if rboxes.shape[0] == 0:
        return rboxes
    quads = np.zeros((rboxes.shape[0], 8), dtype=np.float32)
    for i, rbox in enumerate(rboxes):
        if len(rbox!=0):
            if mode == 'xyxya':
                w = rbox[2] - rbox[0]
                h = rbox[3] - rbox[1]
                x = rbox[0] + 0.5 * w
                y = rbox[1] + 0.5 * h
                theta = rbox[4]
            elif mode == 'xywha':
                x = rbox[0]
                y = rbox[1]
                w = rbox[2]
                h = rbox[3]
                theta = rbox[4]
            quads[i, :] = cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))

    return quads


# Datasets
class RA4(object):
    def __init__(self, root_path):
        self.root_path = root_path 
        self.im_path = osp.join(root_path, 'images')
        self.anno_path = osp.join(root_path, 'labels')  
        self.im_files = glob.glob(osp.join(self.im_path,'*.tif'))
        self.anno_files = glob.glob(osp.join(self.anno_path,'*.txt'))
        self.dist_root = root_path.replace(osp.split(self.root_path)[1], osp.split(self.root_path)[1] + '_augment')
        self.dist_im_dir = osp.join(self.dist_root, 'images')
        self.dist_an_dir = osp.join(self.dist_root, 'labels')
        self.CLASSES = ('1', '2','3','4','5')

        makedir(self.dist_im_dir)
        makedir(self.dist_an_dir)

    def parse_annos(self, label):
        bboxes = []
        classnames = []
        with open(label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls, *bbox = line.strip().split()
                classnames.append(cls)
                bboxes.append([eval(x) for x in bbox])
        return classnames, np.array(bboxes)
    
    def save_labels(self, classnames, bboxes, filename):
        dist_label = osp.join(self.dist_an_dir, filename + '.txt')
        gt = ''
        for cls, bbox in zip(classnames,bboxes.tolist()):
            gt += cls + ' ' + ' '.join(str(i) for i in bbox) + '\n'
        with open(dist_label, 'w') as f:
            f.write(gt)


    def save_ims(self, im, filename):
        dist_im = osp.join(self.dist_im_dir, filename + '.tif')
        cv2.imwrite(dist_im, im)



## dataaug
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


def random_affine(img,  targets=(), degree=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degree=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    
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



if __name__ == "__main__":
    
    root_path = 'D:\Datasets\RAChallenge\Task4\warmup'
    # root_path = 'warmup_augment'
    # root_path = '/data-input/RotationDet/data/RAChallenge/warmup'
    dataset = RA4(root_path)
    
    # augment schedule
    res_cnt, box_areas = category_statistics(dataset.anno_files, dataset.CLASSES)

    # area
    min_bbox_area = min(box_areas)

    scheduler = augment_ratio(res_cnt)
    # augmentation
    pair = tqdm(zip(dataset.im_files, dataset.anno_files))
    for img_file, anno_file in pair:
        pair.set_description("augmentation on{}".format(img_file))
        filename = osp.splitext(osp.split(img_file)[1])[0]
        img = cv2.imread(img_file,1)
        classes, _bboxes = dataset.parse_annos(anno_file)
        cnt = max([scheduler[x] for x in np.unique(classes)])  # 该图像需要生成增强的图像数
        flag = 0  # 已经生成的图像数
        while flag != cnt:
            transform = Augment([   HSV(0.5, 0.5, p=0.),
                                    HorizontalFlip(p=0.5),
                                    VerticalFlip(p=0.5),
                                    Affine(degree=90, translate=0.2, scale=(0.5, 2.0), p=1),
                                    # Noise(0.02, p=0.2),
                                    # Blur(1.3, p=0.5),
                                    ], box_mode = 'xyxyxyxy')
            im, bboxes = transform(img, _bboxes.copy())
            # validation judgement
            area_flag = True
            for box in bboxes:
                if bbox_area(box) < min_bbox_area:
                    area_flag = False 

            cond = (bboxes > 0).all() & \
                    (bboxes[:, [0, 2, 4, 6]] < im.shape[1]).all() & \
                    (bboxes[:, [1, 3, 5, 7]] < im.shape[0]).all() & \
                    area_flag

            if not cond:
                continue
            else:
                flag += 1
                name = filename + '_' + str(flag)
                dataset.save_ims(im, name)
                dataset.save_labels(classes, bboxes, name)