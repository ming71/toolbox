import os
import sys
import cv2
import shutil
import zipfile

from lxml.etree import Element,SubElement,tostring
import xml.etree.ElementTree as ElementTree
from xml.dom.minidom import parseString
import xml.dom.minidom
from PIL import Image

def deal(txt_path,xml_path):
    files=os.listdir(txt_path)
    for file in files:
        filename = os.path.splitext(file)[0]
        sufix = os.path.splitext(file)[1]
        if sufix=='.txt':
            x1,y1,x2,y2,x3,y3,x4,y4, cls_name,conf = readtxt(os.path.join(txt_path, file))
            dst_path = os.path.join(xml_path,filename+'.xml')
            pic_name = filename + '.tif'    
            with open(dst_path,'w') as f:
                writexml(dst_path,pic_name,x1,y1,x2,y2,x3,y3,x4,y4, cls_name,conf)


def readtxt(path):
    with open(path,'r') as f:
        contents=f.read()
        objects=contents.split('\n')
        for i in range(objects.count('')):#去掉空格项
           objects.remove('')
        num=len(objects)#物体的数量
        x1s,y1s,x2s,y2s,x3s,y3s,x4s,y4s, cls_names, confs = [],[],[],[],[],[],[],[],[],[]
        for det in objects:
            infos = det.split(' ')
            assert len(infos) == 10, 'Wrong info loaded in : {}'.format(det)
            x1,y1,x2,y2,x3,y3,x4,y4, cls_name, conf = infos
            x1s.append(x1)
            x2s.append(x2)
            x3s.append(x3)
            x4s.append(x4)
            y1s.append(y1)
            y2s.append(y2)
            y3s.append(y3)
            y4s.append(y4)
            cls_names.append(cls_name)
            confs.append(conf)
        return x1s,y1s,x2s,y2s,x3s,y3s,x4s,y4s, cls_names, confs
    


def readsize(path):  
    img=Image.open(img_path+'/'+path)
    width=img.size[0]
    height=img.size[1]    
    return height,width




#创建xml文件
def writexml(dst_path,pic_name,x1,y1,x2,y2,x3,y3,x4,y4, cls_name,conf): 
    node_root=Element('annotation')
    # source
    node_source = SubElement(node_root,'source')
    SubElement(node_source,'filename').text = "%s" % pic_name
    SubElement(node_source,'origin').text = 'GF2/GF3'
    # research
    node_research = SubElement(node_root,'research')
    SubElement(node_research,'version').text = '4.0'
    SubElement(node_research,'provider').text = "%s" % provider
    SubElement(node_research,'author').text = "%s" % author
    SubElement(node_research,'pluginname').text = "%s" % pluginname
    SubElement(node_research,'pluginclass').text = 'Detection'
    SubElement(node_research,'time').text = '2020-07-2020-11'
    # objects
    node_objects = SubElement(node_root, 'objects')
    nt = len(x1)
    for i in range(nt):
        # objects
        node_object = SubElement(node_objects, 'object')
        SubElement(node_object,'coordinate').text = 'pixel'
        SubElement(node_object,'type').text = 'rectangle'
        SubElement(node_object,'description').text = 'None'
        node_possibleresult = SubElement(node_object,'possibleresult')
        SubElement(node_possibleresult,'name').text = "%s" % cls_name[i]
        SubElement(node_possibleresult,'probability').text = "%s" % conf[i]
        # points
        node_points = SubElement(node_object,'points')
        SubElement(node_points,'point').text = "%s" % ','.join([x1[i],y1[i]])
        SubElement(node_points,'point').text = "%s" % ','.join([x2[i],y2[i]])
        SubElement(node_points,'point').text = "%s" % ','.join([x3[i],y3[i]])
        SubElement(node_points,'point').text = "%s" % ','.join([x4[i],y4[i]])
        SubElement(node_points,'point').text = "%s" % ','.join([x1[i],y1[i]])

    # 将xml格式转化为字符串便于写入
    # 参数含义：encoding为utf-8确保能写入中文；xml_declaration生成第一行的声明
    xml = tostring(node_root,encoding='utf-8', method="xml",xml_declaration=True,pretty_print=True)  
    dom = parseString(xml)
    with open(dst_path, 'wb') as f:
        f.write(xml)
    return 


    

def check_path(det_res, xml_res):
    if not os.path.exists(det_res):
        raise RuntimeError('No detections founded!')
    if  os.path.exists(xml_res):
        shutil.rmtree(xml_res)    
    os.mkdir(xml_res)


 
def zip_dir(dirname,zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(os.path.join(root, name))
        
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        #print arcname
        zf.write(tar,arcname)
    zf.close()


if __name__ == "__main__":
    provider = 'Beijing Institute of Technology'
    author = r'理工大附小'
    pluginname = 'Airplane Detection and Recognition in Optical Images'

    root_dir = r'C:\Users\xiaoming\Desktop'
    det_res = os.path.join(root_dir, 'detections')     # 检测结果按照txt输出到这个文件夹
    xml_res = os.path.join(root_dir, 'xml_temp') 
    submission = os.path.join(root_dir, 'submission.zip')

    check_path(det_res, xml_res)
    deal(det_res,xml_res)      #把txt中的内容写进xml
    zip_dir(xml_res,submission)
    shutil.rmtree(xml_res)


