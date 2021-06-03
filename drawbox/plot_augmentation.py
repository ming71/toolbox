import os
import cv2
import torch
import shutil
import numpy as np

    

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


def plot_gt(img, bboxes, im_name, mode='xyxyxyxy'):

    if  not os.path.exists('temp'):
        os.mkdir('temp')
    img = img.permute(1,2,0)  # transform to HWC for opencv style
    img = np.array(img)[:,:,::-1] if torch.is_tensor(img) else img[:,:,::-1]  # transform to BGR for opencv style

    if bboxes is None:
        cv2.imwrite(os.path.join('temp','test_augment_%s' % (im_name)),img)
    else:
        if mode in ['xywha', 'xyxya']:
            bboxes = rbox_2_quad(bboxes,mode=mode)
        for box in bboxes:
            img = cv2.polylines(cv2.UMat(img),[box.reshape(-1,2).astype(np.int32)],True,(0,0,255),2)
            cv2.imwrite(os.path.join('temp','train_augment_%s' % (im_name)),img)
    print('Check augmentation results in `temp` folder!!!')
