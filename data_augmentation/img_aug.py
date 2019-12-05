import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import math
import imgaug.augmenters as iaa
import torch

from utils.utils import get_rotated_coors


class HSV(object):
    def __init__(self , saturation=0, brightness=0, p=0.5):
        self.saturation = saturation 
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels):
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
    def __init__(self, sigma=0 ,p=0.5):
        self.sigma = sigma 
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0,self.sigma))
            img = blur_aug.augment_image(img)
        return img, labels


class Grayscale(object):
    def __init__(self, grayscale=0. ,p=0.5):
        self.alpha = random.uniform(grayscale,1.0)
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, labels


class Gamma(object):
    def __init__(self, intensity=0 ,p=0.5):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            gm = random.uniform(1-self.intensity,1+self.intensity)
            img = np.uint8(np.power(img/float(np.max(img)), gm)*np.max(img))
        return img, labels


class Noise(object):
    def __init__(self, intensity=0 ,p=0.5):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, labels



class Sharpen(object):
    def __init__(self, intensity=0 ,p=0.5):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity,1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, labels


class Contrast(object):
    def __init__(self, intensity=0 ,p=0.5):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            contrast_aug = aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img=contrast_aug.augment_image(img)
        return img, labels




class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            img = np.fliplr(img)
            if len(labels):
                labels[:, 1] = img.shape[1] - labels[:, 1]
                labels[:, 5] = -labels[:, 5]
        return img, labels        


class VerticalFlip(object):
    def __init__(self ,p=0.5):
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            img = np.flipud(img)
            if len(labels):
                labels[:, 2] = img.shape[0] - labels[:, 2]
                labels[:, 5] = -labels[:, 5]
        return img, labels 


class Affine(object):
    def __init__(self, degrees = 0., translate = 0., scale = 0., shear = 0., p=0.5):
        self.degrees = degrees 
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            img, labels = random_affine(img, labels,
                                        degrees=self.degrees,
                                        translate=self.translate,
                                        scale=self.scale,
                                        shear=self.shear)
        return img, labels 



class CopyPaste(object):
    def __init__(self, mean = 0 , sigma = 0, p=0.5):
        self.mean = mean
        self.sigma = sigma
        self.p = np.clip(p, 0, 0.5)
        # 遵循3sigma原则，在船体侧边一个h位置为mean=0，偏移的范围约束在0+3*sigma内(2*sigma就够了)
        self.pos = abs(np.random.normal(self.mean, self.sigma))


    def __call__(self, img, labels):
        boxes_w = labels[:,3]
        boxes_h = labels[:,4]
        boxes_a = labels[:,5]
        areas = boxes_w * boxes_h
        object_coors = [get_rotated_coors(i).reshape(-1,2).astype(np.int32)  for i in labels[:,1:]]
        pasted_img=img.copy()
        for i,coor in enumerate(object_coors):
            a=boxes_a[i]; w=boxes_w[i]; h=boxes_h[i]; area = areas[i]
            area_ratio = areas[i]/img.shape[0]/img.shape[1]
            # vis验证bbox计算无误
            # img = cv2.polylines(img,[coor],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
            # cv2.imshow('drawbox',img)
            # cv2.moveWindow('drawbox',100,100)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            M_up   = np.float32([[1, 0, -h*(1+self.pos)*np.cos(math.pi*0.5+a)], [0, 1, -h*(1+self.pos)*np.sin(math.pi*0.5+a)]])
            M_down = np.float32([[1, 0,  h*(1+self.pos)*np.cos(math.pi*0.5+a)], [0, 1,  h*(1+self.pos)*np.sin(math.pi*0.5+a)]])
            # 分别获得bbox上下邻域的梯度
            sobel_up  , up_masked_img   , up_pos_mask   = cal_sobel(M_up, coor,img)
            sobel_down, down_masked_img , down_pos_mask = cal_sobel(M_down, coor,img)
            up_masked_img   = cv2.cvtColor(up_masked_img,   cv2.COLOR_BGR2GRAY)
            down_masked_img = cv2.cvtColor(down_masked_img, cv2.COLOR_BGR2GRAY)
            # 获取gt_mask
            gt_mask = np.zeros(img.shape[:-1], np.uint8)
            cv2.fillConvexPoly(gt_mask, coor, (1, 1))

            if  area_ratio<0.01:   self.p *= 1.2  # 小目标尤其丢的厉害，加倍加倍
            # 两侧都不越界，为了考虑海面反光导致的梯度骤增采用作差法。适合场景：陆海/海面/陆地
            if not sobel_up.all() and not sobel_down.all() and random.random() < self.p:	
                grad_diff = ((sobel_up>20).sum()-(sobel_down>20).sum())/area	# thre: 0.1
                pix_diff = abs((up_masked_img).sum()/(area*255) - (down_masked_img).sum()/(area*255)) # 防止模糊图像的梯度都平滑带来误操作
                if grad_diff < 0.15 and pix_diff < 0.15:	# 两侧环境一致，均为海面，两边等概率paste
                    if random.random()<0.7:
                        pasted_img = copy_paste(pasted_img,gt_mask,up_pos_mask)
                        labels = np.row_stack((labels,generate_label(M_up,labels[i])))
                    if random.random()<0.7:
                        pasted_img = copy_paste(pasted_img,gt_mask,down_pos_mask)
                        labels = np.row_stack((labels,generate_label(M_down,labels[i])))
                else: 		# 半海半陆地，选海面paste
                    if up_masked_img.sum()<down_masked_img.sum() : 
                        pos_mask = up_pos_mask
                        M = M_up
                    else:
                        pos_mask = down_pos_mask
                        M = M_down
                    pasted_img = copy_paste(pasted_img,gt_mask,pos_mask)
                    labels = np.row_stack((labels,generate_label(M,labels[i])))
            else:		# 越界增强有风险，没有差分对比，容易误判，暂时不做增强
                pass


        # vis:可视化检查正确性
        # fig = plt.figure(figsize=(10, 10))   
        # ax1 = fig.add_subplot(121)
        # ax1.imshow(img)
        # plt.title('img')
        # plt.axis('off')

        # ax4 = fig.add_subplot(122)
        # ax4.imshow(pasted_img)
        # plt.title('pasted_img')
        # plt.axis('off')
        # plt.show()

        return pasted_img, labels

# 注意labels是 c xywha
class Transform(object):
    # probs设定整体的增强概率，也可以单独再设置，默认均1
    def __init__(self, augmentations, probs = 1):
        self.augmentations = augmentations
        self.probs = probs
        
    def __call__(self, img, labels):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                img, labels = augmentation(img, labels)
        return img, labels



# 一次传入的是一张图像 target是xyxya的格式
# 注意：仅支持box无形变的augment
def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    
    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # # # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)


    M =  T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]

        targets[:,5]-=a/180*math.pi # 逆时针
        targets[:,5][targets[:,5]> 0.5*math.pi] -= math.pi
        targets[:,5][targets[:,5]<-0.5*math.pi] += math.pi

        transcx = targets[:,1] * M[0,0] + targets[:,2] * M[0,1] + M[0,2]
        transcy = targets[:,1] * M[1,0] + targets[:,2] * M[1,1] + M[1,2]
        targets[:,1] = transcx
        targets[:,2] = transcy
        targets[:,[3,4]] *= s

        # reject warped points outside of image
        targets[:,1] = targets[:,1].clip(0, width)
        targets[:,2] = targets[:,2].clip(0, height)
        w = targets[:,3]
        h = targets[:,4]
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        coors = torch.stack([get_rotated_coors(i) for i in torch.from_numpy(targets[:,1:])],0)
        i = (w > 4) & \
            (h > 4) & \
            (ar < 15) & \
            torch.stack([((i[::2] <img.shape[1]) * (i[::2] >0)).all() for i in coors],0).numpy() & \
            torch.stack([((i[1::2]<img.shape[0]) * (i[1::2]>0)).all() for i in coors],0).numpy()
       
        targets = targets[i]
    return imw, targets



def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2, x1y1x2y2=True):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # random mask_size up to 50% image size
    mask_h = random.randint(1, int(h * 0.5))
    mask_w = random.randint(1, int(w * 0.5))

    # box center
    cx = random.randint(0, h)
    cy = random.randint(0, w)

    xmin = max(0, cx - mask_w // 2)
    ymin = max(0, cy - mask_h // 2)
    xmax = min(w, xmin + mask_w)
    ymax = min(h, ymin + mask_h)

    # apply random color mask
    mask_color = [random.randint(0, 255) for _ in range(3)]
    image[ymin:ymax, xmin:xmax] = mask_color

    # return unobscured labels
    if len(labels):
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        labels = labels[ioa < 0.90]  # remove >90% obscured labels
    return labels


##########  copy-paste实现相关
def coor_trans(M,coor):
    tcoor = [0 for i in range(8)]
    coor_x = coor[:,0]
    coor_y = coor[:,1]
    tx = M[0,0]*coor_x + M[0,1]*coor_y + M[0,2]
    ty = M[1,0]*coor_x + M[1,1]*coor_y + M[1,2]
    tcoor[::2] = tx
    tcoor[1::2] = ty
    return np.array(tcoor).reshape(4,2).astype(np.int32)

def cal_sobel(M,coor,img):
    mask = np.zeros(img.shape[:-1], np.uint8)
    tcoor = coor_trans(M,coor)
    # 校验变换
    if (tcoor>0).all() and (tcoor[:,0]<img.shape[1]).all() and (tcoor[:,1]<img.shape[0]).all() :
        cv2.fillConvexPoly(mask, tcoor, (1, 1))
        masked_img = img * np.expand_dims(mask,-1)	# 找到目标区域的mask
        sobel = filter(masked_img)[...,0]	# 三通道，没必要
        pos_mask = mask.copy()
        cv2.fillConvexPoly(pos_mask, tcoor, (1, 1))

        return sobel,masked_img,pos_mask
    else:
        mask.fill(255)
        return mask, img,mask

def filter(img):
	img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	x = cv2.Sobel(img_gry, cv2.CV_16S, 1, 0)
	y = cv2.Sobel(img_gry, cv2.CV_16S, 0, 1)
	xy = cv2.Sobel(img_gry,cv2.CV_16S, 1 , 1)
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	xy = cv2.convertScaleAbs(xy)
	sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
	sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

	return sobel

def copy_paste(img,gt_mask,pos_mask):
    # gt_mask包含paste的内容；pos_mask提供位置（0-1 mask）
	pasted = img.copy() 
	pasted[pos_mask!=0]=img[gt_mask!=0]  
	return pasted 

def generate_label(M,label):
    new_label = label.copy()
    cx = label[1]; cy = label[2]; 
    tx = M[0,0]*cx + M[0,1]*cy + M[0,2]
    ty = M[1,0]*cx + M[1,1]*cy + M[1,2]
    new_label[1] = tx
    new_label[2] = ty
    return new_label




# # ---------------------
# def get_rotated_coors(box):
#         assert len(box) > 0 , 'Input valid box!'
#         cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
#         xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
#         t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
#         R = np.eye(3)
#         R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
#         x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
#         y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
#         x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
#         y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
#         x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
#         y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
#         x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
#         y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 
#         if isinstance(x0,torch.Tensor):
#             r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
#                             x1.unsqueeze(0),y1.unsqueeze(0),
#                             x2.unsqueeze(0),y2.unsqueeze(0),
#                             x3.unsqueeze(0),y3.unsqueeze(0)], 0)
#         else:
#             r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
#         return r_box


# if __name__ == "__main__":
    

#     path = '/py/datasets/HRSC2016/yolo-dataset/train'
#     img_files = os.listdir(path)
#     img_files = [i for i in img_files if i.endswith('jpg')]
#     for img_file in img_files:
#         img = cv2.imread(os.path.join(path,img_file),1)
#         labels = np.loadtxt(os.path.join(path,img_file)[:-4]+'.txt')
#         if len(labels.shape)<2:
#             labels = np.array([labels])
#         labels[:,[1,3]] *= img.shape[1]
#         labels[:,[2,4]] *= img.shape[0]

#         cp = CopyPaste(sigma= 0.1)
#         cp(img,labels)