# date：   2018.12.17 
# update： 2019.9.27
# author:  ming71

# tiny dataset partition from a large one

import os 
import sys
import random
import ipdb
import shutil
random.seed(66666)


def clear_folder(path):
    if os.listdir(path) == []:
        print('{} is already clean'.format(path))
    else:
        files = os.listdir(path)
        for file in files:
            os.remove(os.path.join(path,file))



def division_and_copy(src_path,dst_path,indexes):
    files= os.listdir(src_path) 
    files.sort()        # !!!!排序，不然label就乱了   
    for index in  indexes:
        src = os.path.join(src_path,files[index])
        dst = os.path.join(dst_path,files[index])
        shutil.copyfile(src,dst)



if __name__ == "__main__":
        
    # Setting
    train_ratio      = 0.7     # 数据集不大时，验证集多拿一点
    val_ratio        = 0.3
    tiny_size        = 1000          # tiny数据集大小
    src_imgs         = r'/py/datasets/ship/ships/image'
    src_labels       = r'/py/datasets/ship/ships/label'
    dst_train_imgs   = r'/py/datasets/ship/tiny_ships/source_ships/train_imgs'
    dst_train_labels = r'/py/datasets/ship/tiny_ships/source_ships/train_labels'
    dst_val_imgs     = r'/py/datasets/ship/tiny_ships/source_ships/val_imgs'
    dst_val_labels   = r'/py/datasets/ship/tiny_ships/source_ships/val_labels'

    total_size = len(os.listdir(src_imgs))  # 原始数据集大小

    # 生成随机index
    index = random.sample([i for i in range(total_size)],tiny_size)  
    train_index = random.sample(index,int(tiny_size*train_ratio))
    val_index = [i for i in index if i not in  train_index]

    # 清空目标文件夹
    clear_folder(dst_train_imgs)
    clear_folder(dst_train_labels)
    clear_folder(dst_val_imgs)
    clear_folder(dst_val_labels)

    division_and_copy(src_imgs,dst_train_imgs,train_index)
    division_and_copy(src_imgs,dst_val_imgs,val_index)
    division_and_copy(src_labels,dst_train_labels,train_index)
    division_and_copy(src_labels,dst_val_labels,val_index)


