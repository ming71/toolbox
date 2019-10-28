#-*- coding: UTF-8 -*- 
'''
生成随机数，在数据集中随机采样获得小样本并提取出来
'''
import os
import random
import ipdb
import shutil


# ipdb.set_trace()
# 注意最后的‘/’不可掉了，否则拼接不上
data='/py/datasets/ships/image/'
tiny_data='/py/datasets/tiny_ships/image/'
label='/py/datasets/ships/label/'
tiny_label='/py/datasets/tiny_ships/voc-label/'
val_data='/py/datasets/val_ships/image/'
val_label='/py/datasets/val_ships/voc-label/'


index = random.sample([i for i in range(1,2225)],200)	#1-2225中生成200个随机数list


## 开始复制制作子数据集,分别复制随机的图像和label
#img_index = [str(i).zfill(6)+'.jpg' for i in index]	#转化成图像和label的名称
#label_index = [str(i).zfill(6)+'.xml' for i in index]
#src_index = [data+i for i in img_index]	#加上路径前缀和文件后缀
#src_label_index = [label+i for i in label_index]
#dst_index = [tiny_data+i for i in img_index]
#dst_label_index = [tiny_label+i for i in label_index]
#for i in range(len(index)):
#	shutil.copyfile(src_index[i],dst_index[i])
#	shutil.copyfile(src_label_index[i],dst_label_index[i])



# 上述200张之外的全部作验证集
all_index=[i for i in range(1,2225)]
val_index=[i for i in all_index if i not in index]	#筛选出不在list1(200个)的所有剩下list2(0-2225)元素

val_img_name = [str(i).zfill(6)+'.jpg' for i in val_index]	#转化成图像和label的名称
val_label_name = [str(i).zfill(6)+'.xml' for i in val_index]

src_val_img_path = [data+i for i in val_img_name]	 #加上路径前缀和文件后缀
src_val_label_label_path = [label+i for i in val_label_name]	

dst_val_img_path=[val_data+i for i in val_img_name]
dst_val_label_label_path=[val_label+i for i in val_label_name]

for i in range(len(val_index)):
	shutil.copyfile(src_val_img_path[i],dst_val_img_path[i])
	shutil.copyfile(src_val_label_label_path[i],dst_val_label_label_path[i])





