# ICDAR坐标为四点多边形
# 这里将其处理成近似拟合的矩形框，并且归一化得到yolo格式
# 去除do not care的label

import os
import sys
import cv2
import math
import numpy as np 
from tqdm import tqdm
from decimal import Decimal


# 检查异常文件并返回
# 异常类型：1. xywh数值超出1（图像范围）  2. 负值（max和min标反了的）
def check_exception(txt_path):
   files = os.listdir(txt_path)
   class_id = []
   exception = []
   for file in files:
      with open(os.path.join(txt_path,file),'r') as f:
         contents = f.read()
         lines = contents.split('\n')
         lines = [i for i in lines if len(i)>0]
         for line in lines:
            line = line.split(' ')
            
            assert len(line) == 6 ,'wrong length!!'
            c,x,y,w,h,a = line
            if c not in class_id:
               class_id.append(c)
            if float(x)>1.0 or float(y)>1.0 or float(w)>1.0 or float(h)>1.0 or (float(eval(a))>0.5*math.pi or float(eval(a))<-0.5*math.pi):
               exception.append(file)
            elif float(x)<0 or float(y)<0 or float(w)<0 or float(h)<0:
               exception.append(file)
            
   assert '0' in class_id , 'Class counting from 0 rather than 1!'
   if len(exception) ==0:
      return 'No exception found.'
   else:
      return exception
            
            

def convert(src_path, img_path, dst_path, care_all=False):
   icdar_files= os.listdir(src_path)                            
   for icdar_file in tqdm(icdar_files):                                      #每个文件名称
      with open(os.path.join(dst_path, os.path.splitext(icdar_file)[0]+'.txt'),'w') as f:   #打开要写的文件
         with open(os.path.join(src_path,icdar_file),'r',encoding='utf-8-sig') as fd:        #打开要读的文件
               objects = fd.readlines()
               # objects = [x[ :x.find(x.split(',')[8])-1] for x in objects]
               assert len(objects) > 0, 'No object found in ' + xml_path 

               class_label = 0      # 只分前景背景
               height, width, _ = cv2.imread(os.path.join(img_path, os.path.splitext(icdar_file)[0][3:])+'.jpg').shape

               for object in objects:
                  if care_all:    #  
                     object = object.split(',')[:8]
                     coors = np.array([int(x) for x in object]).reshape(4,2).astype(np.int32)
                     ((cx, cy), (w, h), theta) = cv2.minAreaRect(coors)
                     ###  vis & debug  opencv 0度起点，顺时针为+
                     # print(cv2.minAreaRect(coors))
                     # img = cv2.imread(os.path.join(img_path, os.path.splitext(icdar_file)[0][3:])+'.jpg')
                     # points = cv2.boxPoints(cv2.minAreaRect(coors)).astype(np.int32)
                     # img = cv2.polylines(img,[points],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
                     # cv2.imshow('display box',img)
                     # cv2.waitKey(0)
                     # 转换为自己的标准：-0.5pi, 0.5pi
                     a = theta / 180 * math.pi
                     if a >  0.5*math.pi: a = math.pi - a
                     if a < -0.5*math.pi: a = math.pi + a
                     x = Decimal(cx/width).quantize(Decimal('0.000000'))
                     y = Decimal(cy/height).quantize(Decimal('0.000000'))
                     w = Decimal(w/width).quantize(Decimal('0.000000'))
                     h = Decimal(h/height).quantize(Decimal('0.000000'))
                     a = Decimal(a).quantize(Decimal('0.000000'))
                     f.write(str(class_label)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(a)+'\n')

                  elif '###' not in object :
                     object = object.split(',')[:8]
                     coors = np.array([int(x) for x in object]).reshape(4,2).astype(np.int32)
                     ((cx, cy), (w, h), theta) = cv2.minAreaRect(coors)
                     a = theta / 180 * math.pi
                     if a >  0.5*math.pi: a = math.pi - a
                     if a < -0.5*math.pi: a = math.pi + a
                     x = Decimal(cx/width).quantize(Decimal('0.000000'))
                     y = Decimal(cy/height).quantize(Decimal('0.000000'))
                     w = Decimal(w/width).quantize(Decimal('0.000000'))
                     h = Decimal(h/height).quantize(Decimal('0.000000'))
                     a = Decimal(a).quantize(Decimal('0.000000'))
                     f.write(str(class_label)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(a)+'\n')


if __name__ == "__main__":
   
   care_all = True   # 是否去掉不care的字符
   src_path = "/py/datasets/ICDAR2015/ICDAR/train_labels"
   img_path = '/py/datasets/ICDAR2015/ICDAR/train_imgs'
   dst_path = "/py/datasets/ICDAR2015/yolo/care_all/train"

   convert(src_path, img_path, dst_path, care_all)
   
   exception_files = check_exception(dst_path)
   print(exception_files)
