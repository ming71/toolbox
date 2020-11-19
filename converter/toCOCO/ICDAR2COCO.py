import os
import cv2
import glob
import json
import sys
from PIL import Image

import numpy as np
import dota_utils as util
from utils import *
from tqdm import tqdm

class_name = ['text']

def ICDAR2COCOTrain(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labels')

    data_dict = {}
    info = { 'contributor': 'ming71',
             'data_created': '2020',
             'description': 'ICDAR15 dataset',
             'url': 'sss',
             'version': '1.0',
             'year': 2020}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(labelparent)
        filenames = glob.glob(labelparent + '/*.txt')

        for file in tqdm(filenames):
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.jpg')

            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = parse_icdar_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def ICDAR2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    # labelparent = os.path.join(srcpath, 'labelTxt')
    data_dict = {}
    info = { 'contributor': 'ming71',
             'data_created': '2020',
             'description': 'ICDAR15 dataset',
             'url': 'sss',
             'version': '1.0',
             'year': 2020}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(imageparent)
        filenames = glob.glob(imageparent + '/*.jpg')


        for file in tqdm(filenames):
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.jpg')
            # img = cv2.imread(imagepath)
            img = Image.open(imagepath)
            # height, width, c = img.shape
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)




def parse_icdar_poly(filename):
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r',encoding='UTF-8-sig')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    count = 0
    while True:
        line = f.readline()
        count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(',')
            object_struct = {}
            object_struct['name'] = 'text'
            object_struct['difficult'] = '0' if splitlines[-1] != '###' else '1'
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            poly = list(map(lambda x:np.array(x), object_struct['poly']))
            object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            if (object_struct['long-axis'] < 15):
                object_struct['difficult'] = '1'
                global small_count
                small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects

def parse_icdar_poly2(filename):
    objects = parse_icdar_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects


if __name__ == '__main__':

    ICDAR2COCOTrain(r'/data-input/AerialDetection-master/data/ICDAR15/train',
                   r'/data-input/AerialDetection-master/data/ICDAR15/train/IC15.json',
                   class_name)
    ICDAR2COCOTest(r'/data-input/AerialDetection-master/data/ICDAR15/val',
                  r'/data-input/AerialDetection-master/data/ICDAR15/val/IC15.json',
                  class_name)






