
'''
进阶版：
author: ming71
creat time:2018.12.14
（1）xml不写非目标类：
    扫描class得到name,是不想要的类别时，continue跳出循环，就不会写入xml了
    但是，为了物体数目匹配，每出现一次这种类，物体参数num-=1,避免后面报错
（2）判断是否为非目标图（只含非目标类）：
    扫描每个文件时，cnt变量初始化0
    每个文件在扫描name得到目标类时，cnt+=1，文件最后判断cnt若为0，说明只有非目标类，打印path即可
    
注意：
修改自己的txt类型需要改动的地方：
（1）文件路径
（2）标号对应的class名字定义
（3）xmin等参数的位置，上述23都在readtxt函数中
'''
from lxml.etree import Element,SubElement,tostring
from xml.dom.minidom import parseString
import xml.dom.minidom
import os
import sys
from PIL import Image
import cv2 as cv

#把txt中的内容写进xml
def deal(txt_path,xml_path):
    files=os.listdir(txt_path)#列出所有文件
    for file in files:
        filename=os.path.splitext(file)[0]#分割出所有不带后缀的文件名
        #print(filename)
        #a=input()
        sufix=os.path.splitext(file)[1]#分割出后缀
        #print(sufix)
        #pause=input()
        if sufix=='.txt':
            xmins=[]
            ymins=[]
            xmaxs=[]
            ymaxs=[]
            names=[]
            num,xmins,ymins,xmaxs,ymaxs,names=readtxt(txt_path+'/'+file)
            dealpath=xml_path+"/"+filename+".xml"
            filename=filename+'.jpg'    #别害怕，这里是写入xml的名字而已，还没读图片
            with open(dealpath,'w') as f:
                writexml(dealpath,filename,num,xmins,ymins,xmaxs,ymaxs,names)


#读取txt文件获取五个基本信息
def readtxt(path):
    with open(path,'r') as f:
        contents=f.read()
        #print(contents)        #txt所有内容
        #a=input()
        objects=contents.split('\n')#分割出每个物体
        for i in range(objects.count('')):#去掉空格项
           objects.remove('')
        #print(objects)
        #a=input()
        num=len(objects)#物体的数量
        #print(num)
        #a=input()
        xmins=[]
        ymins=[]
        xmaxs=[]
        ymaxs=[]
        names=[]
        cnt=0
        for objecto in objects:
            #print(objecto)	
            #a=input()
            xmin=objecto.split(' ')[7]  #以','进行分割，得到的片段存入列表，[n]代表列表的第n个分片
            #print(objecto.split(' '))	
            #a=input()
            xmin=xmin.strip()	#去除首尾空格
            #print(xmin)
            #a=input()

            ymin=objecto.split(' ')[8]
            ymin=ymin.strip()
            #print(ymin)
            #a=input()
			
            xmax=objecto.split(' ')[9]
            xmax=xmax.strip()
            #print(xmax)
            #a=input()
			
            ymax=objecto.split(' ')[10]
            ymax=ymax.strip()
            #print(ymax)
            #a=input()
			
            name=objecto.split(' ')[4]
            name=name.strip()
            #print(name)	
            #a=input()
#此处自定义名字			
            if name=="1":
                name='carrier'
                cnt+=1
            elif (name=="2" or name=='4'):
                name='cruiser'
                cnt+=1
            elif (name== "5" or name=='6'):
                name='ship'
                cnt+=1
            else:
                name='?????????????????????????????????'
                #print(path)   
                num=num-1          #目标=物体数-1
                continue	
            if  cnt==0:
                print(path)		#打印出需要删的非目标图
            #print(xmin,ymin,xmax,ymax,name)
            #a=input()
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
            names.append(name)
        #print(num,xmins,ymins,xmaxs,ymaxs,names)
        #a=input()
        #print(cnt)
        return num,xmins,ymins,xmaxs,ymaxs,names
    

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
def writexml(path,filename,num,xmins,ymins,xmaxs,ymaxs,names,height='256',width='256'):    
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
        node_name.text = '%s' % names[i]
        node_name = SubElement(node_object, 'pose')
        node_name.text = '%s' % "unspecified"
        node_name = SubElement(node_object, 'truncated')
        node_name.text = '%s' % "0"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')     
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s'% xmins[i]
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % ymins[i]
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % xmaxs[i]
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % ymaxs[i]

    xml = tostring(node_root, pretty_print=True)  
    dom = parseString(xml)
    with open(path, 'wb') as f:
        f.write(xml)
    return



if __name__ == "__main__":
    txt_path= (r'D:\研究生\第一年\dataset\trainsetnewplus\txt')
    img_path= (r'D:\研究生\第一年\dataset\trainsetnewplus\image')
    xml_path= (r'D:\研究生\第一年\dataset\trainsetnewplus\label')
    deal(txt_path,xml_path)      #把txt中的内容写进xml
    dealwh(img_path,xml_path)
    
