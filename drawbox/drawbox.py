import numpy as np
import cv2
import os
import ipdb
import math

# pts=np.array([[203,708],[211,704],[229,720],[221,727]])
# stable:
#   object_coors.append(np.array([[int(x0),int(y0)],[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)]]))

# 功能：
# 本部分自定义，输入单个xml文件，返回文件中所有物体的bbox坐标点的list
# list下每个元素为一个物体，是numpy数组
# return的四个点顺时针逆时针都可以，但是别调对角线


def get_HRSCplus_points(label_path,rotate=False):
    with open(label_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<object>')	
        objects.pop(0)

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in objects:
            if not rotate:
                xmin = object[object.find('<xmin>')+10 : object.find('</xmin>')]
                ymin = object[object.find('<ymin>')+10 : object.find('</ymin>')]
                xmax = object[object.find('<xmax>')+10 : object.find('</xmax>')]
                ymax = object[object.find('<ymax>')+10 : object.find('</ymax>')]
                x0=xmin; y0=ymin; x1=xmin; y1=ymax; x2=xmax; y2=ymax; x3=xmax; y3=ymin
            else:   # 旋转框
                cx = float(object[object.find('<rbox_cx>')+9 : object.find('</rbox_cx>')])
                cy = float(object[object.find('<rbox_cy>')+9 : object.find('</rbox_cy>')])
                w  = float(object[object.find('<rbox_w>')+8 : object.find('</rbox_w>')])
                h  = float(object[object.find('<rbox_h>')+8 : object.find('</rbox_h>')])
                a  = object[object.find('<rbox_ang>')+10 : object.find('</rbox_ang>')]

                xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
                t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
                a = float(a) if not a[0]=='-' else -float(a[1:])
                # print(-a*180/math.pi)
                R = np.eye(3)
                R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
                x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
                y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
                x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
                y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
                x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
                y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
                x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
                y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors 



# for xml style(四点八坐标)
def get_xml_points(label_path, rotate=False):
    with open(label_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<object>')	
        objects.pop(0)

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
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors  


# for HRSC style(xml标注)
# 两种模式：rect是两点的xyxy格式； rotate是xywha(其中a顺时针为正，+-0.5pi,x+为0)
def get_HRSC_points(label_path,rotate=False):
    with open(label_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<HRSC_Object>')	
        objects.pop(0)

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in objects:
            if not rotate:
                xmin = object[object.find('<box_xmin>')+10 : object.find('</box_xmin>')]
                ymin = object[object.find('<box_ymin>')+10 : object.find('</box_ymin>')]
                xmax = object[object.find('<box_xmax>')+10 : object.find('</box_xmax>')]
                ymax = object[object.find('<box_ymax>')+10 : object.find('</box_ymax>')]
                x0=xmin; y0=ymin; x1=xmin; y1=ymax; x2=xmax; y2=ymax; x3=xmax; y3=ymin
            else:   # 旋转框
                cx = float(object[object.find('<mbox_cx>')+9 : object.find('</mbox_cx>')])
                cy = float(object[object.find('<mbox_cy>')+9 : object.find('</mbox_cy>')])
                w  = float(object[object.find('<mbox_w>')+8 : object.find('</mbox_w>')])
                h  = float(object[object.find('<mbox_h>')+8 : object.find('</mbox_h>')])
                a  = object[object.find('<mbox_ang>')+10 : object.find('</mbox_ang>')]

                xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
                t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
                a = float(a) if not a[0]=='-' else -float(a[1:])
                # print(-a*180/math.pi)
                R = np.eye(3)
                R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
                x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
                y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
                x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
                y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
                x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
                y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
                x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
                y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors  


# for DOTA style（四点八坐标txt格式）
def get_DOTA_points(label_path, rotate=False):
    with open(label_path,'r') as f:        
        contents=f.read()
        lines=contents.split('\n')
        lines = [x for x in contents.split('\n')  if x]	 # 移除空格

        lines = lines[2:]    # 移除信息行

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in lines:
            coors = object.split(' ')
            x0 = coors[0]; y0 = coors[1]; x1 = coors[2]; y1 = coors[3]
            x2 = coors[4]; y2 = coors[5]; x3 = coors[6]; y3 = coors[7]
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors  



# for ICDAR style（四点八坐标txt格式）
def get_ICDAR_points(label_path,rotate=False):
    with open(label_path,'r',encoding='UTF-8-sig') as f:        
        contents=f.read()
        lines=contents.split('\n')
        lines = [x for x in contents.split('\n')  if x]	 # 移除空格

        object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
        for object in lines:
            coors = object.split(',')
            x0 = coors[0]; y0 = coors[1]; x1 = coors[2]; y1 = coors[3]
            x2 = coors[4]; y2 = coors[5]; x3 = coors[6]; y3 = coors[7]
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors  



# for yolo style（归一化的xywh，txt格式）
# yolo比较特殊，文件夹下有img有txt label，需要区分筛选
# 为了简单，图像格式设置jpg，可手动更改
# 旋转的格式是cxywha，仍归一化，其中a是顺时针为正，+-0.5pi
def get_yolo_points(img_path, rotate=False):
    filename = os.path.split(img_path)[1]
    if filename.endswith('.jpg'):
        label_path = os.path.join(os.path.split(img_path)[0], filename[:-4]+'.txt')
        height,width,_ = cv2.imread(img_path).shape
        with open(label_path,'r',encoding="utf8", errors='ignore') as f:        
            contents=f.read()
            lines=contents.split('\n')
            lines = [x for x in contents.split('\n')  if x]	 # 移除空格

            object_coors=[]	# coor内一个元素是一个物体，含四个点8坐标
            for object in lines:
                coors = object.split(' ')
                x = float(coors[1])*width
                y = float(coors[2])*height
                w = float(coors[3])*width
                h = float(coors[4])*height
                if not rotate:
                    x0 = x-0.5*w; y0 = y-0.5*h; x1 = x+0.5*w; y1 = y-0.5*h
                    x2 = x+0.5*w; y2 = y+0.5*h; x3 = x-0.5*w; y3 = y+0.5*h
                else:       ## 旋转
                    a = coors[5]
                    print([x,y,w,h,a])
                    xmin = x - w*0.5; xmax = x + w*0.5; ymin = y - h*0.5; ymax = y + h*0.5
                    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
                    a = float(a) if not a[0]=='-' else -float(a[1:])
                    R = np.eye(3)
                    R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(x,y), scale=1)
                    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
                    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
                    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
                    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
                    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
                    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
                    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
                    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 
                object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
        return object_coors  
    else:
        return []


# 输入：单张图片和其标注xml提取的所有物体坐标的list,save_flag
# coor格式：array([[ 43, 436],[103, 496],[472, 127],[412,  67]]
def drawbox(img_path,object_coors,save_flag=False,save_path=None):
    print(img_path)
    img=cv2.imread(img_path,1)
    for coor in object_coors:
        img = cv2.polylines(img,[coor],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
        if save_flag:
        	cv2.imwrite(os.path.join(save_path,os.path.split(img_path)[1]), img)
        else: 
            cv2.imshow(img_path,img)
            cv2.moveWindow(img_path,100,100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # img_path = '/py/datasets/HRSC2016/yolo-dataset/train'
    # label_path = '/py/datasets/HRSC2016/yolo-dataset/train'
    # save_path = '/py/datasets/ship/tiny_ships/yolo_ship/single_class/train_imgs'
    
    # 注意：如果是yolo，img和label放到一个文件夹下，并且路径设置也要一样
    img_path = '/py/BoxesCascade/ICDAR15/train_img/img_34.jpg'
    label_path = '/py/BoxesCascade/ICDAR15/train_label/gt_img_34.txt' #34
    func = get_ICDAR_points
    
    
    # for folder
    if os.path.isdir(img_path) and os.path.isdir(label_path):
        img_files = os.listdir(img_path)
        xml_files = os.listdir(label_path)
        img_files.sort()	# 进行排序，使得xml和img序列一致可以zip
        xml_files.sort()
        iterations = zip(img_files,xml_files)
        for iter in iterations:
            if func != get_ICDAR_points:
                assert os.path.splitext(iter[0])[0]==os.path.splitext(iter[1])[0],'图像和label不匹配！'
            # 选择label的模式
            # object_coors = get_yolo_points(os.path.join(label_path,iter[1]), rotate=True)
            if not iter[0].endswith('.txt'):
                object_coors = func(os.path.join(label_path,iter[1]),True)
                if len(object_coors):
                    drawbox(os.path.join(img_path,iter[0]),object_coors)
                else:
                    print('No obj!')
    
    # for single img
    elif os.path.isfile(label_path):
        object_coors = func(os.path.join(label_path),rotate=False)
        if len(object_coors):
            drawbox(img_path,object_coors,False)
    else:
        print('Path Not Matched!!!')
