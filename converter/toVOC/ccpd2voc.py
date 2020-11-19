#encoding:utf-8
'''
author :        Captain
creat time :    2019.06.25
function description：
转换CCPD的GT格式到VOC的GT格式
'''
from lxml.etree import Element,SubElement,tostring
from xml.dom.minidom import parseString
import xml.dom.minidom
import os
import sys
from PIL import Image

# 把txt中的内容写进xml
def deal(img_folder, xml_path):
    files = os.listdir(img_folder) # 列出所有文件
    for file in files:
        print('file: ', file)
        iname = file.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        print('iname: ', iname)
        filename = os.path.splitext(file)[0] # 分割出所有不带后缀的文件名
        print('filename: ', filename)
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        print(leftUp, rightDown)
        num, xmins, ymins, xmaxs, ymaxs, names = 1, leftUp[0], leftUp[1], rightDown[0], rightDown[1], 'plate'
        dealpath = xml_path + "/" + filename + ".xml"
        filename = filename + '.jpg' # 别害怕，这里是写入xml的名字而已，还没读图片
        with open(dealpath,'w') as f:
            writexml(dealpath, filename, num, xmins, ymins, xmaxs, ymaxs, names, 1160, 720)

'''
#读取图片的高和宽写入xml
def dealwh(img_path, xml_path):
    files = os.listdir(img_path)#列出所有文件
    for file in files:
        filename = os.path.splitext(file)[0]#分割出文件名
        #print(filename)
        #a=input()
        sufix = os.path.splitext(file)[1]#分割出后缀
        #print(sufix)
        #a=input()
        if sufix=='.jpg':       #在这里改图片类型
            height,width=readsize(img_path+'/'+file)
            #print(height,width)
            #a=input()
            dealpath = xml_path+"/"+filename+".xml"
            #print(dealpath)
            #a=input()
            gxml(dealpath,height,width)     #在xml文件中添加宽和高信息

def readsize(path):
    img=Image.open(path)
    width=img.size[0]
    height=img.size[1]
    return height,width

#在xml文件中添加宽和高
def gxml(path,height,width):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    heights = root.getElementsByTagName('height')[0]
    heights.firstChild.data = height
    #print(height)

    widths = root.getElementsByTagName('width')[0]
    widths.firstChild.data = width
    #print(width)
#    filename = root.getElementsByTagName('filename')[0]
#    print(filename.firstChild.data)
    with open(path, 'w') as f:
        dom.writexml(f)
    return
'''

#创建xml文件
def writexml(path, filename, num, xmins, ymins, xmaxs, ymaxs, names, height='1160', width='720'):
    node_root=Element('annotation')

    node_folder=SubElement(node_root,'folder')
    node_folder.text="CCPDdevkit"

    node_filename=SubElement(node_root,'filename')
    node_filename.text="%s" % filename

    node_size=SubElement(node_root,"size")
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = '%s' % names
    node_name = SubElement(node_object, 'pose')
    node_name.text = '%s' % "Unspecified"
    node_name = SubElement(node_object, 'truncated')
    node_name.text = '%s' % "0"
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')

    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = '%s'% xmins
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = '%s' % ymins
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = '%s' % xmaxs
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = '%s' % ymaxs

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    with open(path, 'wb') as f:
        f.write(xml)
    return

if __name__ == "__main__":
    img_folder = '/home/dup/Downloads/CCPD2019/ccpd_base' # 要处理的CCPD文件目录
    xml_folder = '/home/dup/work/opensource/ssd/data/CCPDdevkit/Annotations' # 处理得到的真值文件
    deal(img_folder, xml_folder) # 把txt中的内容写进xml
#    dealwh(img_folder, xml_folder) # 把w,h的内容写进xml
