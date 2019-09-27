import numpy as np
import cv2
import os
import ipdb


#pts=np.array([[203,708],[211,704],[229,720],[221,727]])

# 功能：本部分自定义，输入单个xml文件，返回文件中所有物体的bbox坐标点的list
# list下每个元素为一个物体，是numpy数组

# for xml style(四点八坐标)
def get_xml_points(label_path):
    with open(label_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<object>')	
        objects.pop(0)
        assert len(objects) > 0, 'No object found in ' + label_path 

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in objects:
            x0 = object[object.find('<xmin>')+6 : object.find('</xmin>')]
            y0 = object[object.find('<ymin>')+6 : object.find('</ymin>')]
            x1 = object[object.find('<xmin>')+6 : object.find('</xmin>')]
            y1 = object[object.find('<ymax>')+6 : object.find('</ymax>')]
            x2 = object[object.find('<xmax>')+6 : object.find('</xmax>')]
            y2 = object[object.find('<ymax>')+6 : object.find('</ymax>')]
            x3 = object[object.find('<xmax>')+6 : object.find('</xmax>')]
            y3 = object[object.find('<ymin>')+6 : object.find('</ymin>')]
            object_coors.append(np.array([[int(x0),int(y0)],[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)]]))
    return object_coors  


# for DOTA style（四点八坐标txt格式）
def get_DOTA_points(label_path):
    with open(label_path,'r') as f:        
        contents=f.read()
        # ipdb.set_trace()
        lines=contents.split('\n')
        lines = [x for x in contents.split('\n')  if x]	 # 移除空格

        lines = lines[2:]    # 移除信息行
        assert len(lines) > 0, 'No object found in ' + label_path 

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in lines:
            coors = object.split(' ')
            x0 = coors[0]
            y0 = coors[1]
            x1 = coors[2]
            y1 = coors[3]
            x2 = coors[4]
            y2 = coors[5]
            x3 = coors[6]
            y3 = coors[7]
            object_coors.append(np.array([[int(x0),int(y0)],[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)]]))
    return object_coors  

# for yolo style（归一化的xywh，txt格式）
def get_yolo_points(label_path,img_file):
    height,width,_ = cv2.imread(img_file).shape
    with open(label_path,'r') as f:        
        contents=f.read()
        # ipdb.set_trace()
        lines=contents.split('\n')
        lines = [x for x in contents.split('\n')  if x]	 # 移除空格

        assert len(lines) > 0, 'No object found in ' + label_path 

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in lines:
            coors = object.split(' ')
            x = float(coors[1])*width
            y = float(coors[2])*height
            w = float(coors[3])*width
            h = float(coors[4])*height
            x0 = x-0.5*w
            y0 = y-0.5*h
            x1 = x+0.5*w
            y1 = y-0.5*h
            x2 = x+0.5*w
            y2 = y+0.5*h
            x3 = x-0.5*w
            y3 = y+0.5*h
            object_coors.append(np.array([[int(x0),int(y0)],[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)]]))
    return object_coors  



# 输入：单张图片和其标注xml提取的所有物体坐标的list,save_flag
def drawbox(img_path,object_coors,save_flag=False,save_path=None):
    print(img_path)
    img=cv2.imread(img_path,1)
    for coor in object_coors:
        img = cv2.polylines(img,[coor],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
        if save_flag:
        	cv2.imwrite(os.path.join(save_path,os.path.split(img_path)[1]), img)
        else: 
            cv2.imshow('drawbox',img)
    cv2.waitKey(0)


if __name__ == "__main__":
    img_path = '/py/datasets/ship/tiny_ships/source_ships/train_imgs/001691.jpg'
    label_path = '/py/datasets/ship/tiny_ships/yolo_ship/train_yolo_anno/001691.txt'
    save_path = '/py/toolbox/DOTA标注'
    # ipdb.set_trace()
    # for folder
    if os.path.isdir(img_path) and os.path.isdir(label_path):
        img_files = os.listdir(img_path)
        xml_files = os.listdir(label_path)
        img_files.sort()	# 进行排序，使得xml和img序列一致可以zip
        xml_files.sort()
        iterations = zip(img_files,xml_files)
        for iter in iterations:
        	# object_coors = get_points(os.path.join(label_path,iter[1]))
        	object_coors = get_yolo_points(os.path.join(label_path,iter[1]),img_file=os.path.join(img_path,iter[0]))
        	drawbox(os.path.join(img_path,iter[0]),object_coors,False,save_path=save_path)
    # for single img
    elif os.path.isfile(label_path):
        # pass
        object_coors = get_yolo_points(label_path,img_path)
        drawbox(img_path,object_coors,False)
    else:
        print('Path Not Matched!!!')
