'''
replace the raw format with the aimed format imgs
'''

import os
import cv2
from PIL import Image
 
# 将目标文件夹下所有 指定后缀的文件 全部另存为目标格式
def format_trans(src_suffix,dst_suffix,path,save_path):
	for i,filename in enumerate(os.listdir(path)):
		# print(filename)
		if os.path.splitext(filename)[1] == src_suffix:
			img = Image.open(path+"/"+filename)
			img = img.convert('RGB')  # 丢弃透明度通道避免转换出错
			img.save(save_path + '/' +  filename.strip(src_suffix) + dst_suffix)
		print('\r Converting  {:.2f}%'.format(100*i/len(os.listdir(path))),end='')




if __name__ == '__main__':
	path = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOC2007/JPEGImages-png'
	save_path = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOC2007/JPEGImages'
	src_suffix = '.png'
	dst_suffix = '.jpg'

	format_trans(src_suffix,dst_suffix,path,save_path)
