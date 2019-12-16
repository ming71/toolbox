# date：   2018.12.17 
# update： 2019.12.04
# author:  ming71

# dataset partition

import os 
import sys
import random
import ipdb
import shutil
# random.seed(666)


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
    train_ratio      = 0.6     # 数据集不大时，验证集测试集多拿一点
    val_ratio        = 0.2
    test_ratio       = 0.2
    src_imgs         = r'/py/datasets/HRSC+/extra/img'
    src_labels       = r'/py/datasets/HRSC+/extra/label'
    dst_train_imgs   = r'/py/datasets/HRSC+/train'
    dst_train_labels = r'/py/datasets/HRSC+/train_label'
    dst_val_imgs     = r'/py/datasets/HRSC+/val'
    dst_val_labels   = r'/py/datasets/HRSC+/val_label'    
    dst_test_imgs     = r'/py/datasets/HRSC+/test'
    dst_test_labels   = r'/py/datasets/HRSC+/test_label'

    total_size = len(os.listdir(src_imgs))  # 原始数据集大小

    # 生成随机index
    index = set([i for i in range(total_size)])
    train_index = set(random.sample(index,int(total_size*train_ratio)))
    val_index   = set(random.sample(index-train_index,int(total_size*val_ratio)))
    test_index = index-train_index-val_index

    # 清空目标文件夹
    clear_folder(dst_train_imgs)
    clear_folder(dst_train_labels)
    clear_folder(dst_val_imgs)
    clear_folder(dst_val_labels)
    clear_folder(dst_test_imgs)
    clear_folder(dst_test_labels)

    division_and_copy(src_imgs,dst_train_imgs,train_index)
    division_and_copy(src_imgs,dst_val_imgs,val_index)
    division_and_copy(src_imgs,dst_test_imgs,test_index)
    division_and_copy(src_labels,dst_train_labels,train_index)
    division_and_copy(src_labels,dst_val_labels,val_index)
    division_and_copy(src_labels,dst_test_labels,test_index)
