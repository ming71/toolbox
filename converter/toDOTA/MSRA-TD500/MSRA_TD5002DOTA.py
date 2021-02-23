import os
import sys
import glob
import math
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm


def rotate(theta, x, y):
    rotatex = math.cos(theta) * x - math.sin(theta) * y
    rotatey = math.cos(theta) * y + math.sin(theta) * x
    return rotatex, rotatey
 
def xy_rorate(theta, x, y, cx, cy):
    r_x, r_y = rotate(theta, x - cx, y - cy)
    return cx + r_x, cy + r_y

def rec_rotate(x, y, w, h, t):
    cx = x + w / 2
    cy = y + h / 2
    x1, y1 = xy_rorate(t, x, y, cx, cy)
    x2, y2 = xy_rorate(t, x + w, y, cx, cy)
    x3, y3 = xy_rorate(t, x, y + h, cx, cy)
    x4, y4 = xy_rorate(t, x + w, y + h, cx, cy)
    return x1, y1,  x3, y3, x4, y4,x2, y2
 

def convert_label(gt_path, dst_path):
    f = open(gt_path,'r')
    savestr = ''
    for line in f:
        _, diffcult, lx, ly, w, h, theta  = [eval(x) for x in line.strip().split(' ')]
        quads = ' '.join([str(round(x))  for x in rec_rotate(lx, ly, w, h, theta)])
        quads += (' text ' + str(diffcult) + '\n')
        savestr += quads
    savef = open(dst_path,'w')
    savef.write(savestr)
    savef.close()


def generate_txt_labels(root_path):
    label_dir = osp.join(root_path, 'labels')
    label_txt_path = osp.join(root_path, 'labelTxt')
    if  osp.exists(label_txt_path):
        shutil.rmtree(label_txt_path)
    os.mkdir(label_txt_path)

    label_paths = glob.glob(os.path.join(label_dir, '*.txt'))
    pbar = tqdm(label_paths)
    for label in pbar:
        pbar.set_description("MSRA_TD500 generate_txt in {}".format(root_path))
        label_txt = label.replace('labels', 'labelTxt')
        convert_label(label, label_txt)

