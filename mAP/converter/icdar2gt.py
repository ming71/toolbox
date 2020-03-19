import os 
import cv2
import numpy as np
import shutil
from decimal import Decimal

# 注意文件结构！dets和gt的目录是map测试文件所在的目录下
# 由于文本的不重叠性，直接在所有result加上conf = 0.99

def convert_detections(det_path):
    eval_dets = os.path.join(os.path.split(det_path)[0], 'detection-results')
    if os.path.exists(eval_dets):
        shutil.rmtree(eval_dets)
    os.mkdir(eval_dets)
    dets = os.listdir(det_path)
    for det in dets:
        with open(os.path.join(det_path, det), 'r') as f:
            res = f.readlines()
            res = ''.join(['text 0.99 '+x.replace(',', ' ') for x in res])
            with open(os.path.join(eval_dets, det[4:]), 'w') as fd: # 标准格式是res_1_img.txt所以从4开始
                fd.write(res)

def convert_gt(gt_path):
    eval_gts = os.path.join(os.path.split(gt_path)[0], 'ground-truth')
    if os.path.exists(eval_gts):
        shutil.rmtree(eval_gts)
    os.mkdir(eval_gts)
    gts = os.listdir(gt_path)
    for gt in gts:
        with open(os.path.join(gt_path, gt), 'r') as f:
            res = f.readlines()
            diff = ['###' in x for x in res]    # 标记###的作为difficult不被计入
            res = ''.join(['text ' + ' '.join(x.split(',')[:8]) + '\n' if not diff[i] \
                else 'text ' + ' '.join(x.split(',')[:8]) + ' difficult' + '\n'  \
                for i, x in enumerate(res)])
            with open(os.path.join(eval_gts, gt[3:]), 'w') as fd:
                fd.write(res)


if __name__ == "__main__":
    
    det_path = '/py/BoxesCascade/test/dets'     # 检测输出结果文件夹
    gt_path = '/py/BoxesCascade/test/gts' # 给定的gt文件夹

    convert_detections(det_path)
    convert_gt(gt_path)

    

