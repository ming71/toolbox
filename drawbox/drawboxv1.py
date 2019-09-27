import numpy as np
import cv2
import os
import ipdb


## ---------------简单demo-------
# path =  r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/img_src/P0000_1536_2560.png'
# pts=np.array([[203,708],[211,704],[229,720],[221,727]])

# def drawbox(path):
#     img=cv2.imread(path,1)
#     img = cv2.polylines(img,[pts],True,(0,0,255),2)
#     cv2.imshow('drawbox',img)
#     cv2.waitKey(0)
	
# if __name__ == "__main__":
#     drawbox(path)


# ----------实际用的程序---------------
pts=np.array([[203,708],[211,704],[229,720],[221,727]])

# 功能：本部分自定义，输入单个xml文件，返回文件中所有物体的bbox坐标点的list
# list下每个元素为一个物体，是numpy数组
def get_points(xml_path):
    with open(xml_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<object>')	
        objects.pop(0)
        assert len(objects) > 0, 'No object found in ' + xml_path 

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in objects:
		# ---------------- for DOTA ----------------------
            x0 = object[object.find('<x0>')+4 : object.find('</x0>')]
            y0 = object[object.find('<y0>')+4 : object.find('</y0>')]
            x1 = object[object.find('<x1>')+4 : object.find('</x1>')]
            y1 = object[object.find('<y1>')+4 : object.find('</y1>')]
            x2 = object[object.find('<x2>')+4 : object.find('</x2>')]
            y2 = object[object.find('<y2>')+4 : object.find('</y2>')]
            x3 = object[object.find('<x3>')+4 : object.find('</x3>')]
            y3 = object[object.find('<y3>')+4 : object.find('</y3>')]
		# ---------------- for VOC --------------------
#             x0 = object[object.find('<xmin>')+6 : object.find('</xmin>')]
#             y0 = object[object.find('<ymin>')+6 : object.find('</ymin>')]
#             x1 = object[object.find('<xmin>')+6 : object.find('</xmin>')]
#             y1 = object[object.find('<ymax>')+6 : object.find('</ymax>')]
#             x2 = object[object.find('<xmax>')+6 : object.find('</xmax>')]
#             y2 = object[object.find('<ymax>')+6 : object.find('</ymax>')]
#             x3 = object[object.find('<xmax>')+6 : object.find('</xmax>')]
#             y3 = object[object.find('<ymin>')+6 : object.find('</ymin>')]
            object_coors.append(np.array([[int(x0),int(y0)],[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)]]))
    return object_coors  


# 输入：单张图片和其标注xml提取的所有物体坐标的list,save_flag
def drawbox(img_path,object_coors,save_flag=False,save_path=None):
    img=cv2.imread(img_path,1)
    for coor in object_coors:
        img = cv2.polylines(img,[coor],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
        if save_flag:
        	cv2.imwrite(save_path, img)
        cv2.imshow('drawbox',img)
    cv2.waitKey(0)

if __name__ == "__main__":
    img_path = '/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/img_dst'
    xml_path = '/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_dst'
    save_path = '/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug'
    if os.path.isdir(img_path) and os.path.isdir(img_path):
        img_files = os.listdir(img_path)
        xml_files = os.listdir(xml_path)
        img_files.sort()	# 进行排序，使得xml和img序列一致可以zip
        xml_files.sort()
        iterations = zip(img_files,xml_files)
        for iter in iterations:
        	object_coors = get_points(os.path.join(xml_path,iter[1]))
        	drawbox(os.path.join(img_path,iter[0]),object_coors)
    elif os.path.isfile(xml_path):
        object_coors = get_points(xml_path)
        drawbox(img_path,object_coors,False)
    else:
        print('Path Not Matched!!!')

    

