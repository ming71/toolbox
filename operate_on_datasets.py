''''
ming71
2019.10.20

支持的功能:
	- 数据集校验和匹配清除
	- 大型数据集的复制
'''


from shutil import copyfile
import os
from tqdm import tqdm
import glob

# label和img匹配校验,并根据最小数目的label或者img剔除多余的部分
def check(imgs_path,labels_path,force=False):
	imgs   = sorted(os.listdir(imgs_path))
	labels = sorted(os.listdir(labels_path))
	img_names   = [os.path.splitext(name)[0] for name in imgs]
	label_names = [os.path.splitext(name)[0] for name in labels]
	if img_names == label_names:
		print('Matched OK!')
	elif force == True:
		difference = list(set(img_names) ^ set(label_names))
		for diff in tqdm(difference):
			if diff in img_names:
				id = img_names.index(diff)
				os.remove(os.path.join(imgs_path,imgs[id]))
			if diff in label_names:
				id = label_names.index(diff)
				os.remove(os.path.join(labels_path,labels[id]))
	else:
		difference = list(set(img_names) ^ set(label_names))	# 不同时出现在ab中的
		print('Not matched items:\n{}'.format(difference))



# 文件复制(当前不支持递归复制,文件夹下不能包含子文件夹)
def copy(src,dst):
	src_files = os.listdir(src)
	for src_file in tqdm(src_files):
		src_file = os.path.join(src,src_file)					  # 绝对路径
		dst_file = os.path.join(dst,os.path.split(src_file)[1])   # 绝对路径
		copyfile(src_file,dst_file)

if __name__ == "__main__":
	# src = '/py/datasets/COCO2014/labels/val2014'
	# dst = '/media/bit530/ming/env/dataset/COCO2014/yolo-coco/labels/val2014'
	# copy(src,dst)

	imgs = '/py/datasets/COCO2014/train2014'
	labels = '/py/datasets/COCO2014/labels/train2014'
	check(imgs,labels,force=True)