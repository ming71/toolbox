import os
import sys



##--------------- 直接按照文件夹生成 -----------------
img_path = r"/data-input/das/DOTA/trainsplit/images" 
set_file = r'/data-input/das/DOTA/train.txt'


files= os.listdir(img_path) #得到文件夹下的所有文件名称
with open(set_file,'w') as f:
	for file in files:
		filename, extension = os.path.splitext(file)
		if extension in ['.jpg', '.bmp','.png']:
			f.write(os.path.join(img_path,file)+'\n')




## --------------- 按照VOC的imageset文件名生成 --------------- 
# trainset = r'/data-input/das/HRSC2016/ImageSets/train.txt'
# valset   = r'/data-input/das/HRSC2016/ImageSets/val.txt'
# testset  = r'/data-input/das/HRSC2016/ImageSets/test.txt'
# img_dir = r'/data-input/das/HRSC2016/FullDataSet/AllImages'
# label_dir = r'/data-input/das/HRSC2016/FullDataSet/Annotations'
# root_dir = r'/data-input/das/HRSC2016'  # 存放生成带绝对路径的txt imgset文件的路径


# for dataset in [trainset, valset, testset]:
#     with open(dataset,'r') as f:
#         names = f.readlines()
#         paths = [os.path.join(img_dir,x[:-1]+'.jpg\n') for x in names]
#         with open(os.path.join(root_dir,os.path.split(dataset)[1]), 'w') as fw:
#             fw.write(''.join(paths))

