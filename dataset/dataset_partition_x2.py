#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os 
import sys
import random
import ipdb
import shutil
from tqdm import tqdm
random.seed(666)


def clear_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.listdir(path) == []:
        print('{} is already clean'.format(path))
    else:
        files = os.listdir(path)
        for file in files:
            os.remove(os.path.join(path,file))



def division_and_copy(src_path,dst_path,indexes):
    files= os.listdir(src_path) 
    files.sort()        # !!!!排序，不然label就乱了   
    for index in tqdm(indexes):
        src = os.path.join(src_path,files[index])
        dst = os.path.join(dst_path,files[index])
        shutil.copyfile(src,dst)



if __name__ == "__main__":
        
    # Setting
    train_ratio      = 0.8    
    val_ratio        = 0.2
    src_imgs         = r'total_dataset/images'
    src_labels       = r'total_dataset/labelTxt'
    dst_train_imgs   = r'train/images'
    dst_train_labels = r'train/labelTxt'
    dst_val_imgs     = r'val/images'
    dst_val_labels   = r'val/labelTxt'  

    total_size = len(os.listdir(src_imgs))  # 原始数据集大小

    # 生成随机index
    index = set([i for i in range(total_size)])
    train_index = set(random.sample(index,int(total_size*train_ratio)))
    val_index   = set(random.sample(index-train_index,int(total_size*val_ratio)))

    # 清空目标文件夹
    clear_folder(dst_train_imgs)
    clear_folder(dst_train_labels)
    clear_folder(dst_val_imgs)
    clear_folder(dst_val_labels)

    print('copying train imgs...')
    division_and_copy(src_imgs,dst_train_imgs,train_index)
    print('copying val imgs...')
    division_and_copy(src_imgs,dst_val_imgs,val_index)
    print('copying train labels...')
    division_and_copy(src_labels,dst_train_labels,train_index)
    print('copying val labels...')
    division_and_copy(src_labels,dst_val_labels,val_index)
