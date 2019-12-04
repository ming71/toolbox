import os
import sys
import cv2
import math
from decimal import Decimal

# 检查异常文件并返回
# 异常类型：1. xywh数值超出1（图像范围）  2. 负值（max和min标反了的）
def check_exception(txt_path,rotate=False):
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
            if not rotate:
               assert len(line) == 5 ,'wrong length!!'
               c,x,y,w,h = line
               if c not in class_id:
                  class_id.append(c)
               if float(x)>1.0 or float(y)>1.0 or float(w)>1.0 or float(h)>1.0:
                  exception.append(file)
               elif float(x)<0 or float(y)<0 or float(w)<0 or float(h)<0:
                  exception.append(file)
            else: 
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
            
            

def convert(xml_path,txt_path,class_names):
   xml_files= os.listdir(xml_path)                            
   for xml_file in xml_files:                                      #每个文件名称
      with open(os.path.join(txt_path,os.path.splitext(xml_file)[0]+'.txt'),'w') as f:   #打开要写的文件
         with open(os.path.join(xml_path,xml_file),'r') as fd:        #打开要读的文件
               contents=fd.read()
               objects=contents.split('<object>')
               assert len(objects) > 0, 'No object found in ' + xml_path 
               
               info = objects.pop(0)
               width  = int(info[info.find('<width>')+7 : info.find('</width>')])
               height = int(info[info.find('<height>')+8 : info.find('</height>')])

               for object in objects:
                  # 提取类别id (class根据需要自定义)
                  class_name = object[object.find('<name>')+6 : object.find('</name>')]
                  class_label = class_names.index(class_name)
                  # 提取坐标
                  xmin = int(object[object.find('<xmin>')+6 : object.find('</xmin>')])
                  xmax = int(object[object.find('<xmax>')+6 : object.find('</xmax>')])
                  ymin = int(object[object.find('<ymin>')+6 : object.find('</ymin>')])
                  ymax = int(object[object.find('<ymax>')+6 : object.find('</ymax>')])
                  x = Decimal(0.5*(xmax+xmin)/width).quantize(Decimal('0.000000'))
                  y = Decimal(0.5*(ymax+ymin)/height).quantize(Decimal('0.000000'))
                  w = Decimal((xmax-xmin)/width).quantize(Decimal('0.000000'))
                  h = Decimal((ymax-ymin)/height).quantize(Decimal('0.000000'))

                  f.write(str(class_label)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')

def convert_HRSCplus(xml_path,txt_path,class_names,rotate=True):
   xml_files= os.listdir(xml_path)                            
   for xml_file in xml_files:                                      #每个文件名称
      with open(os.path.join(txt_path,os.path.splitext(xml_file)[0]+'.txt'),'w') as f:   #打开要写的文件
         with open(os.path.join(xml_path,xml_file),'r') as fd:        #打开要读的文件
               contents=fd.read()
               objects=contents.split('<object>')
               assert len(objects) > 0, 'No object found in ' + xml_path 
               
               info = objects.pop(0)
               width  = int(info[info.find('<width>')+7 : info.find('</width>')])
               height = int(info[info.find('<height>')+8 : info.find('</height>')])

               for object in objects:
                  # 提取类别id (class根据需要自定义)
                  class_name = object[object.find('<name>')+6 : object.find('</name>')]
                  class_label = 0
                  # 提取坐标
                  if not rotate:
                     xmin = int(object[object.find('<xmin>')+6 : object.find('</xmin>')])
                     xmax = int(object[object.find('<xmax>')+6 : object.find('</xmax>')])
                     ymin = int(object[object.find('<ymin>')+6 : object.find('</ymin>')])
                     ymax = int(object[object.find('<ymax>')+6 : object.find('</ymax>')])
                     x = Decimal(0.5*(xmax+xmin)/width).quantize(Decimal('0.000000'))
                     y = Decimal(0.5*(ymax+ymin)/height).quantize(Decimal('0.000000'))
                     w = Decimal((xmax-xmin)/width).quantize(Decimal('0.000000'))
                     h = Decimal((ymax-ymin)/height).quantize(Decimal('0.000000'))
                     f.write(str(class_label)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')

                  else:
                     cx = float(object[object.find('<rbox_cx>')+9 : object.find('</rbox_cx>')])
                     cy = float(object[object.find('<rbox_cy>')+9 : object.find('</rbox_cy>')])
                     w  = float(object[object.find('<rbox_w>')+8 : object.find('</rbox_w>')])
                     h  = float(object[object.find('<rbox_h>')+8 : object.find('</rbox_h>')])
                     a  = object[object.find('<rbox_ang>')+10 : object.find('</rbox_ang>')]
                     cx = Decimal(cx/width).quantize(Decimal('0.000000'))
                     cy = Decimal(cy/height).quantize(Decimal('0.000000'))
                     w = Decimal(w/width).quantize(Decimal('0.000000'))
                     h = Decimal(h/height).quantize(Decimal('0.000000'))
                     a = Decimal(a).quantize(Decimal('0.000000'))
                     f.write(str(class_label)+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+' '+str(a)+'\n')


if __name__ == "__main__":
   xml_path = r"/py/datasets/HRSC+/partition/train_label"
   txt_path =  r"/py/datasets/HRSC+/partition/train_yolo"

   class_names = ['ship']  # 从0开始编号

   convert_HRSCplus(xml_path,txt_path,class_names,rotate=True)
   
   exception_files = check_exception(txt_path,rotate=True)
   print(exception_files)
