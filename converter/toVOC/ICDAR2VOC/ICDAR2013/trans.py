#! /usr/bin/python  
#coding:utf-8  
  
import os, sys
import shutil
import glob  
from PIL import Image  

# generate VOC2007 dir
cur_path = os.getcwd()
Annotations = cur_path + '\\' + 'VOC2007' + '\\' + 'Annotations'
os.makedirs(Annotations)
Main = cur_path + '\\' + 'VOC2007' + '\\' + 'ImageSets' + '\\' + 'Main'
os.makedirs(Main)
open('trainval.txt','w+').close()
shutil.move('trainval.txt',Main)
open('test.txt','w+').close()
shutil.move('test.txt',Main)
JPEGImages = cur_path + '\\' + 'VOC2007' + '\\' + 'JPEGImages'
os.makedirs(JPEGImages)

# target dir  
base_dir = "VOC2007"  
  
target_img_dir = base_dir + "/" + "JPEGImages"  
target_ann_dir = base_dir + "/" + "Annotations"  
target_set_dir = base_dir + "/" + "ImageSets"  
  
# source train dir  
train_img_dir = "Challenge2_Training_Task12_Images"  
train_txt_dir = "Challenge2_Training_Task1_GT"  
  
  
# source test dir  
test_img_dir = "Challenge2_Test_Task12_Images"
test_txt_dir = "Challenge2_Test_Task1_GT"


# rename and move img to target_img_dir
# train img
for file in os.listdir(train_img_dir):
    os.rename(os.path.join(train_img_dir,file),os.path.join(target_img_dir,"ICDAR2013_Train_" + os.path.basename(file)))  

# test img
for file in os.listdir(test_img_dir):
    os.rename(os.path.join(test_img_dir,file),os.path.join(target_img_dir,"ICDAR2013_Test_" + os.path.basename(file)))  

# create annotations to target_ann_dir
img_Lists = glob.glob(target_img_dir + '/*.jpg')

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))

img_names = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
  
for img in img_names:
    im = Image.open((target_img_dir + '/' + img + '.jpg'))
    width, height = im.size

    # open the crospronding txt file
    if 'Train' in img:
        gt = open(train_txt_dir + '/gt_' + img.split('_')[-1] + '.txt').read().splitlines()
        # write in xml file
        xml_file = open((target_ann_dir + '/' + img + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of text on xml file
        for img_each_label in gt:
            spt = img_each_label.split(' ')
            xml_file.write('    <object>\n')
            xml_file.write('        <name>text</name>\n')  
            xml_file.write('        <pose>Unspecified</pose>\n')  
            xml_file.write('        <truncated>0</truncated>\n')  
            xml_file.write('        <difficult>0</difficult>\n')  
            xml_file.write('        <bndbox>\n')  
            xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')  
            xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')  
            xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')  
            xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')  
            xml_file.write('        </bndbox>\n')  
            xml_file.write('    </object>\n')  
  
        xml_file.write('</annotation>')  
  
    if 'Test' in img:  
        gt = open(test_txt_dir + '/gt_img_' + img.split('_')[-1] + '.txt').read().splitlines()  
    # write in xml file  
        xml_file = open((target_ann_dir + '/' + img + '.xml'), 'w')  
        xml_file.write('<annotation>\n')  
        xml_file.write('    <folder>VOC2007</folder>\n')  
        xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')  
        xml_file.write('    <size>\n')  
        xml_file.write('        <width>' + str(width) + '</width>\n')  
        xml_file.write('        <height>' + str(height) + '</height>\n')  
        xml_file.write('        <depth>3</depth>\n')  
        xml_file.write('    </size>\n')  
  
        # write the region of text on xml file  
        for img_each_label in gt:  
            spt = img_each_label.split(',')  
            xml_file.write('    <object>\n')  
            xml_file.write('        <name>text</name>\n')  
            xml_file.write('        <pose>Unspecified</pose>\n')  
            xml_file.write('        <truncated>0</truncated>\n')  
            xml_file.write('        <difficult>0</difficult>\n')  
            xml_file.write('        <bndbox>\n')  
            xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')  
            xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')  
            xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')  
            xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')  
            xml_file.write('        </bndbox>\n')  
            xml_file.write('    </object>\n')  
  
        xml_file.write('</annotation>')  
  
# write info into target_set_dir  
img_lists = glob.glob(target_ann_dir + '/*.xml')  
img_names = []  
for item in img_lists:  
    temp1, temp2 = os.path.splitext(os.path.basename(item))  
    img_names.append(temp1)  
  
train_fd = open(target_set_dir + "/Main/trainval.txt", 'w')  
test_fd = open(target_set_dir + "/Main/test.txt", 'w')  
  
for item in img_names:  
    if 'Train' in item:  
        train_fd.write(str(item) + '\n')  
    if 'Test' in item:  
        test_fd.write(str(item) + '\n')  