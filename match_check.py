import cv2
import ipdb
import os

#  检查数据集的img和label文件是否一一对应


def laebl_default_check(img_path): 
	# 遍历所有img，检查没有label的image
	label_default = []
	imgs= os.listdir(img_path)
	for i,img in enumerate(imgs):
		print('\r Searching  {:.2f}%'.format(100*i/len(imgs)),end='')
		if img.strip(img_postfix)+label_postfix not in imgs:
			label_default.append(img)
	if not len(label_default):
		print('\n\n---label_default--')
		print(label_default)
	else :
		print('\nNo label default.')


def img_default_check(label_path):
	# 遍历所有label，检查没有image的label
	img_default = []
	labels= os.listdir(label_path)
	for i,label in enumerate(labels):
		print('\r Searching  {:.2f}%'.format(100*i/len(labels)),end='')
		if label.strip(label_postfix)+img_postfix not in labels:
			img_default.append(label)
	if not len(img_default):
		print('\n\n---img_default--')
		print(img_default)
	else :
		print('\nNo img default.')




if __name__ == '__main__':

	img_path = '/py/R2CNN-tensorflow/data/VOCdevkit/VOC2007/JPEGImages'
	label_path = '/py/R2CNN-tensorflow/data/VOCdevkit/VOC2007/Annotations'
	img_postfix = '.png'
	label_postfix = '.xml'

	img_default_check(img_path)
	laebl_default_check(label_path)