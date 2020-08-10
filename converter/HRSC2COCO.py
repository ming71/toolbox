import dota_utils as util
import os
import sys
import cv2
import math
import json
import numpy as np
from utils import *

from PIL import Image
from tqdm import tqdm


L1_names = ['ship']
# TODO: finish them
L2_names = []
L3_names = []

def HRSC2COCOTrain(srcpath, destfile, cls_names, train_set_file):
    imageparent = os.path.join(srcpath, 'FullDataSet/AllImages')
    labelparent = os.path.join(srcpath, 'FullDataSet/Annotations')

    data_dict = {}
    info = { 'contributor': 'ming71',
             'data_created': '2020',
             'description': 'HRSCDataset',
             'url': 'sss',
             'version': '1.0',
             'year': 2016}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(train_set_file, 'r') as f_set:
        filenames = [x.strip('\n') for x in f_set.readlines()]
        # import ipdb;ipdb.set_trace()

    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(labelparent)
        pbar = tqdm(filenames)
        for filename in pbar:
            pbar.set_description("HRSC2COCOTrain")
            imagepath = os.path.join(imageparent, filename + '.jpg')

            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = filename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = parse_voc_poly2(os.path.join(labelparent, filename+'.xml'))
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

def HRSC2COCOTest(srcpath, destfile, cls_names, test_set_file):
    imageparent = os.path.join(srcpath, 'FullDataSet/AllImages')
    data_dict = {}
    info = { 'contributor': 'ming71',
             'data_created': '2020',
             'description': 'HRSCDataset',
             'url': 'sss',
             'version': '1.0',
             'year': 2016}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
    
    with open(test_set_file, 'r') as f_set:
        filenames = [x.strip('\n') for x in f_set.readlines()]

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(imageparent)
        pbar = tqdm(filenames)
        for filename in pbar:
            pbar.set_description("HRSC2COCOTest")
            # image_id = int(basename[1:])
            imagepath = os.path.join(imageparent, filename + '.jpg')
            # img = cv2.imread(imagepath)
            img = Image.open(imagepath)
            # height, width, c = img.shape
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = filename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)



def parse_voc_poly(filename):
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    content = f.read()
    content_splt = [x for x in content.split('<HRSC_Object>')[1:] if x!='']
    count = len(content_splt)
    if count > 0:
        for obj in content_splt:
            object_struct = {}
            object_struct['name'] = 'ship'
            object_struct['difficult'] = '0' 
            cx = float(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')])
            cy = float(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')])
            w  = float(obj[obj.find('<mbox_w>')+8 : obj.find('</mbox_w>')])
            h  = float(obj[obj.find('<mbox_h>')+8 : obj.find('</mbox_h>')])
            a  = obj[obj.find('<mbox_ang>')+10 : obj.find('</mbox_ang>')]
            a = float(a) if not a[0]=='-' else -float(a[1:])
            theta = a*180/math.pi
            points = cv2.boxPoints(((cx,cy),(w,h),theta))
            object_struct['poly'] = [points[0],points[1],points[2],points[3]]
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
        print('No object founded in %s' % filename)
    return objects

def parse_voc_poly2(filename):
    objects = parse_voc_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects


if __name__ == '__main__':

    # HRSC2COCOTrain(r'/data-input/AerialDetection-master/data/HRSC2016',
    #                r'/data-input/AerialDetection-master/data/HRSC2016/train.json',
    #                L1_names,
    #                r'/data-input/AerialDetection-master/data/HRSC2016/ImageSets/train.txt')
    # HRSC2COCOTrain(r'/data-input/AerialDetection-master/data/HRSC2016',
    #                r'/data-input/AerialDetection-master/data/HRSC2016/val.json',
    #                L1_names,
    #                r'/data-input/AerialDetection-master/data/HRSC2016/ImageSets/val.txt')
    # HRSC2COCOTrain(r'/data-input/AerialDetection-master/data/HRSC2016',
    #                r'/data-input/AerialDetection-master/data/HRSC2016/trainval.json',
    #                L1_names,
    #                r'/data-input/AerialDetection-master/data/HRSC2016/ImageSets/trainval.txt')
    HRSC2COCOTest(r'/data-input/AerialDetection-master/data/HRSC2016',
                  r'/data-input/AerialDetection-master/data/HRSC2016/test.json',
                  L1_names,
                  r'/data-input/AerialDetection-master/data/HRSC2016/ImageSets/test.txt')





