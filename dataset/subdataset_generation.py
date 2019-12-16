# 为了方便所有数据集通用，以及简单起见，此处不对yolo作单独处理，默认都是img和label分离不同文件夹
# 如果是yolo，先用op_on_dataset tool将其选择性复制到不同的文件夹再划分子集


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
    sub_ratio        = 0.1
    src_imgs         = '/py/datasets/ICDAR2015/yolo/13+15/val_img'
    src_labels       = '/py/datasets/ICDAR2015/yolo/13+15/val_label'
    dst_sub_imgs   = '/py/datasets/ICDAR2015/yolo/subdata/val'
    dst_sub_labels = '/py/datasets/ICDAR2015/yolo/subdata/val'

    total_size = len(os.listdir(src_imgs))  # 原始数据集大小

    # 生成随机index
    index = set([i for i in range(total_size)])
    sub_index = set(random.sample(index,int(total_size*sub_ratio)))

    # 清空目标文件夹
    clear_folder(dst_sub_imgs)
    clear_folder(dst_sub_labels)

    division_and_copy(src_imgs,dst_sub_imgs,sub_index)
    division_and_copy(src_labels,dst_sub_labels,sub_index)
