'''
replace the raw images with aimed format images
'''

import os
import cv2
import sys
import numpy as np
from PIL import Image
 
path = r'/media/xiaoming/ming/env/dataset/ship_dataset/hsrc'
 
for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.jpg':
        img = Image.open(path+"/"+filename)
        img.save(path + '/' +  filename)
'''     
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        # print(fileName)
        newFileName = fileName[0:fileName.find("_")]+".jpg"
        print(newFileName)
        im = Image.open(file_path+"\\"+fileName)
        im.save(file_path+"\\"+newFileName)
'''
