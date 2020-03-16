from lxml.etree import Element,SubElement,tostring
from xml.dom.minidom import parseString
import xml.dom.minidom
import os
import sys
from PIL import Image
import cv2 as cv
from tqdm import tqdm

#把txt中的内容写进xml
def deal(txt_path,xml_path):
    files=os.listdir(txt_path)#列出所有文件
    for file in tqdm(files):
        filename=os.path.splitext(file)[0]#分割出所有不带后缀的文件名
        #print(filename)
        #a=input()
        sufix=os.path.splitext(file)[1]#分割出后缀
        #print(sufix)
        #pause=input()
        if sufix=='.txt':
            x1=[]
            y1=[]
            x2=[]
            y2=[]            
            x3=[]
            y3=[]
            x4=[]
            y4=[]
            names=[]
            num,x1,y1,x2,y2,x3,y3,x4,y4 = readtxt(txt_path+'/'+file)
            dealpath=xml_path+"/"+filename[3:]+".xml"
            filename=filename+'.jpg'    #别害怕，这里是写入xml的名字而已，还没读图片
            with open(dealpath,'w') as f:
                writexml(dealpath,filename,num,x1,y1,x2,y2,x3,y3,x4,y4)


#读取txt文件获取xy等基本信息
def readtxt(path):
    with open(path,'r',encoding='utf-8-sig') as f:
        contents=f.read()
        objects=contents.split('\n')#分割出每个物体
        for i in range(objects.count('')):#去掉空格项
           objects.remove('')
        num=len(objects)#物体的数量
        x1=[]
        y1=[]
        x2=[]
        y2=[]            
        x3=[]
        y3=[]
        x4=[]
        y4=[]
        for object in objects:
            info = object.split(',')
            x1.append(info[0])
            y1.append(info[1])
            x2.append(info[2])
            y2.append(info[3])
            x3.append(info[4])
            y3.append(info[5])
            x4.append(info[6])
            y4.append(info[7])
        return num,x1,y1,x2,y2,x3,y3,x4,y4
    

#读取图片的高和宽写入xml        
def dealwh(img_path,xml_path):
    files=os.listdir(img_path)#列出所有文件
    for file in files:
        filename=os.path.splitext(file)[0]#分割出文件名
        #print(filename)
        #a=input()
        sufix=os.path.splitext(file)[1]#分割出后缀
        #print(sufix)
        #a=input()
        if sufix=='.jpg':       #在这里改图片类型
            height,width=readsize(file)
            #print(height,width)
            #a=input()
            dealpath=xml_path+"/"+filename+".xml"   
            #print(dealpath)
            #a=input()
            gxml(dealpath,height,width)     #在xml文件中添加宽和高信息


def readsize(path):  
    img=Image.open(img_path+'/'+path)
    width=img.size[0]
    height=img.size[1]    
    return height,width


#在xml文件中添加宽和高
def gxml(path,height,width):
    dom=xml.dom.minidom.parse(path)
    root=dom.documentElement
    heights=root.getElementsByTagName('height')[0]
    heights.firstChild.data=height
    #print(height)

    widths=root.getElementsByTagName('width')[0]
    widths.firstChild.data=width
    #print(width)
    with open(path, 'w') as f:
        dom.writexml(f)
    return

#创建xml文件
def writexml(path,filename,num,x1,y1,x2,y2,x3,y3,x4,y4,name='text',height='256',width='256'):    
    node_root=Element('annotation')

    node_folder=SubElement(node_root,'folder')
    node_folder.text="VOC2007"

    node_filename=SubElement(node_root,'filename')
    node_filename.text="%s" % filename

    node_size=SubElement(node_root,"size")
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(num):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '%s' % name
        node_name = SubElement(node_object, 'pose')
        node_name.text = '%s' % "unspecified"
        node_name = SubElement(node_object, 'truncated')
        node_name.text = '%s' % "0"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')     
        node_x1 = SubElement(node_bndbox, 'x1')
        node_x1.text = '%s'% x1[i]
        node_y1 = SubElement(node_bndbox, 'y1')
        node_y1.text = '%s'% y1[i]
        node_x2 = SubElement(node_bndbox, 'x2')
        node_x2.text = '%s'% x2[i]
        node_y2 = SubElement(node_bndbox, 'y2')
        node_y2.text = '%s'% y2[i]
        node_x3 = SubElement(node_bndbox, 'x3')
        node_x3.text = '%s'% x3[i]
        node_y3 = SubElement(node_bndbox, 'y3')
        node_y3.text = '%s'% y3[i]
        node_x4 = SubElement(node_bndbox, 'x4')
        node_x4.text = '%s'% x4[i]
        node_y4 = SubElement(node_bndbox, 'y4')
        node_y4.text = '%s'% y4[i]                

    xml = tostring(node_root, pretty_print=True)  
    dom = parseString(xml)
    with open(path, 'wb') as f:
        f.write(xml)
    return



if __name__ == "__main__":
    txt_path= (r'/py/stela-master/ICDAR15/train_label')
    img_path= (r'/py/stela-master/ICDAR15/train_img')
    xml_path= (r'/py/stela-master/ICDAR15/train_xml')
    deal(txt_path,xml_path)      #把txt中的内容写进xml
    dealwh(img_path,xml_path)
    
