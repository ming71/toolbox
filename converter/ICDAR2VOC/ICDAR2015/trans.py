#coding:utf-8  
  
import os, sys  
import glob
import shutil
from PIL import Image  
import cv2  
import numpy as np  
import codecs

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
  
target_img_dir = base_dir + "/" + "JPEGImages/"  
target_ann_dir = base_dir + "/" + "Annotations/"  
target_set_dir = base_dir + "/" + "ImageSets/"  
  
# source train dir  
train_img_dir = "ch4_training_images/"  
train_txt_dir = "ch4_training_localization_transcription_gt/"  
  
test_img_dir = "ch4_test_images"  
  
# rename and move img to target_img_dir  
# train img   
  
for file in os.listdir(train_img_dir):  
    os.rename(os.path.join(train_img_dir,file),os.path.join(target_img_dir,"ICDAR2015_Train_" + os.path.basename(file)))  
  
for file in os.listdir(test_img_dir):  
    os.rename(os.path.join(test_img_dir,file),os.path.join(target_img_dir,"ICDAR2015_Test_" + os.path.basename(file)))  
  
img_list = []      
  
for file_name in os.listdir(target_img_dir):  
    img_list.append(file_name)  
  
  
for idx in range(len(img_list)):  
    img_name = target_img_dir + img_list[idx]  
    gt_name = train_txt_dir + 'gt_img_' + img_list[idx].split('.')[0].split('_')[3]+'.txt'  
  
    # print gt_name  
    # gt_split = []
    # with open(gt_name, 'rb') as f:
    #     for line in f:
    #         gt_txt = line.decode("utf-8")
    #         gt_split.append(gt_txt)
    # gt_split = gt_txt.split('\n')
    img = cv2.imread(img_name)
    im = Image.open(img_name)    
    imgwidth, imgheight = im.size

    # write in xml file
    xml_file = open((target_ann_dir + img_list[idx].split('.')[0] + '.xml'), 'w')  
    xml_file.write('<annotation>\n')  
    xml_file.write('    <folder>VOC2007</folder>\n')  
    xml_file.write('    <filename>' + img_list[idx] + '</filename>\n')  
    xml_file.write('    <size>\n')  
    xml_file.write('        <width>' + str(imgwidth) + '</width>\n')  
    xml_file.write('        <height>' + str(imgheight) + '</height>\n')  
    xml_file.write('        <depth>3</depth>\n')  
    xml_file.write('    </size>\n')  
  
    f = False  
    difficult = 0
    with open(gt_name, 'rb') as f:
        for line in f:
            gt_txt = line.decode("utf-8-sig")
            gt_ind = gt_txt.split(',')
            if len(gt_ind) == 9:
                pt1 = (int(gt_ind[0]), int(gt_ind[1]))  
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))  
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))  
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))  
                dtxt = gt_ind[8]  
                if "###" in dtxt:  
                    difficult = 1  
                else:  
                    difficult = 0  
          
                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))  
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))  
  
                angle = 0  
          
                if edge1 > edge2:  
              
                    width = edge1  
                    height = edge2  
                    if pt1[0] - pt2[0] != 0:  
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180  
                    else:  
                        angle = 90.0  
                elif edge2 >= edge1:  
                    width = edge2  
                    height = edge1  
                    #print pt2[0], pt3[0]  
                    if pt2[0] - pt3[0] != 0:  
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180  
                    else:  
                        angle = 90.0  
                if angle < -45.0:  
                    angle = angle + 180  
  
                x_ctr = float(pt1[0] + pt3[0]) / 2#pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2  
                y_ctr = float(pt1[1] + pt3[1]) / 2#pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2  
  
          
  
                # write the region of text on xml file  
                xml_file.write('    <object>\n')  
                xml_file.write('        <name>text</name>\n')  
                xml_file.write('        <pose>Unspecified</pose>\n')  
                xml_file.write('        <truncated>0</truncated>\n')  
                xml_file.write('        <difficult>' + str(difficult) + '</difficult>\n')  
                xml_file.write('        <bndbox>\n')  
                xml_file.write('            <x>' + str(x_ctr) + '</x>\n')  
                xml_file.write('            <y>' + str(y_ctr) + '</y>\n')  
                xml_file.write('            <w>' + str(width) + '</w>\n')  
                xml_file.write('            <h>' + str(height) + '</h>\n')  
                xml_file.write('            <theta>' + str(angle) + '</theta>\n')  
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
for item in img_names:  
    train_fd.write(str(item) + '\n')  
