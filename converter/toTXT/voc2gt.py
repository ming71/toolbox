import os 
import cv2
import numpy as np
import shutil
import math
from tqdm import tqdm
from decimal import Decimal
import xml.etree.ElementTree as ET




def convert_voc_gt(gt_path, dst_path, eval_difficult= False):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    with open(gt_path,'r') as f:
        label_dir = gt_path.replace('/ImageSets/Main/test.txt','/Annotations')
        files = [os.path.join(label_dir, x.strip('\n')+'.xml') for x in f.readlines()]
    gts = [os.path.split(x)[1] for x in files]
    dst_gt = [os.path.join(dst_path, x.replace('.xml','.txt')) for x in gts]
    print('gt generating...')
    for i, filename in enumerate(tqdm(files)):
        tree = ET.parse(filename)
        objs = tree.findall('object')
        boxes, gt_classes = [], []
        nt = 0
        for _, obj in enumerate(objs):
            nt +=1 
            diffculty = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            pt = [
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymax').text),
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymax').text),
            ]
            name = obj.find('name').text.lower().strip()
            with open(dst_gt[i],'a') as fd:
                if eval_difficult:   # 评估的时候带上difficult
                    fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                        name,pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7]
                    ))                        
                else:
                    if diffculty == 0:
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                            name,pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7]
                        ))
                    elif diffculty == 1:
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} difficult\n'.format(
                            name,pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7]
                        ))
                    else:
                        raise RuntimeError('???? difficult wrong!!')                    
        if nt == 0:
            open(dst_gt[i],'w').close()
            # os.remove(filename)
            # os.remove(filename.replace('Annotations','AllImages').replace('xml','jpg'))



if __name__ == "__main__":
    gt_path = '/data-input/das_dota/VOC2007/ImageSets/Main/test.txt' # 给定的测试imgset文件
    dst_path = '/data-input/das_dota/datasets/evaluate/ground-truth'

    eval_difficult = False
    convert_voc_gt(gt_path, dst_path, eval_difficult)

    

