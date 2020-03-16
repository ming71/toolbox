import os
import sys



##--------------- 分类标注 -----------------
img_path = r"/py/BoxesCascade/HRSC2016/Train/AllImages" 
set_file = r'/py/BoxesCascade/HRSC2016/trainval.txt'


files= os.listdir(img_path) #得到文件夹下的所有文件名称
with open(set_file,'w') as f:
	for file in files:
		filename, extension = os.path.splitext(file)
		if extension in ['.jpg', '.bmp']:
			f.write(os.path.join(img_path,file)+'\n')

