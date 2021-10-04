import cv2
import glob
import numpy as np
import os.path as osp
from ops.tools import makedir

from lxml import etree
from lxml.etree import Element,SubElement,tostring
import xml.etree.ElementTree as ElementTree
from xml.dom.minidom import parseString
import xml.dom.minidom


class DOTA(object):
    def __init__(self, root_path):
        self.root_path = root_path 
        self.im_path = osp.join(root_path, 'images')
        self.anno_path = osp.join(root_path, 'labelTxt')  
        self.im_files = glob.glob(osp.join(self.im_path,'*.png'))
        self.anno_files = glob.glob(osp.join(self.anno_path,'*.txt'))
        self.dist_root = root_path.replace(osp.split(self.root_path)[1], osp.split(self.root_path)[1] + '_augment')
        self.dist_im_dir = osp.join(self.dist_root, 'images')
        self.dist_an_dir = osp.join(self.dist_root, 'labelTxt')

        makedir(self.dist_im_dir)
        makedir(self.dist_an_dir)

    def parse_annos(self, label):
        bboxes = []
        classnames = []
        with open(label, 'r') as f:
            lines = f.readlines()[2:]
            for line in lines:
                *bbox, cls, diff = line.strip().split()
                classnames.append(cls)
                bboxes.append([eval(x) for x in bbox])
        return classnames, np.array(bboxes)
    
    
    def save_ims(self, im, filename):
        dist_im = osp.join(self.dist_im_dir, filename + '.png')
        cv2.imwrite(dist_im, im)

    def save_labels(self, classnames, bboxes, size, filename):
        dist_label = osp.join(self.dist_an_dir, filename + '.txt')
        gt = ''
        for cls, bbox in zip(classnames,bboxes.tolist()):
            gt += cls + ' ' + ' '.join(str(i) for i in bbox) + '\n'
        with open(dist_label, 'w') as f:
            f.write(gt)

    def save(self, im, classnames, bboxes, filename):
        self.save_ims(im, filename)
        self.save_labels(classnames, bboxes, im.shape, filename)
