import os
import sys



##--------------- 分类标注 -----------------
img_path = r"/py/datasets/ship/tiny_ships/yolo_ship/val_imgs" 
set_file = r'/py/datasets/ship/tiny_ships/yolo_ship/val.txt'

mode = 'yolo'

files= os.listdir(img_path) #得到文件夹下的所有文件名称

if mode == 'voc':
	with open(set_file,'w') as f:
		for file in files:
			f.write(file[:-4]+'\n')

if mode == 'yolo':
	with open(set_file,'w') as f:
		for file in files:
			filename, extension = os.path.splitext(file)
			if extension == '.jpg':
				f.write(os.path.join(img_path,file)+'\n')
