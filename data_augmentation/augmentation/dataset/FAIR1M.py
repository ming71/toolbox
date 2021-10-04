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

CLASS_NAMES = ['C919',
        'ARJ21',
        'Tractor',
        'Roundabout',
        'Trailer',
        'Passenger Ship',
        'Warship',
        'Football Field',
        'Truck Tractor',
        'Excavator',
        'Bus',
        'Bridge',
        'Baseball Field',
        'A350',
        'Basketball Court',
        'Engineering Ship',
        'Boeing777',
        'Tugboat',
        'A330',
        'Boeing787',
        'Boeing747',
        'other-ship',
        'A321',
        'Tennis Court',
        'Liquid Cargo Ship',
        'other-vehicle',
        'Boeing737',
        'Fishing Boat',
        'A220',
        'Intersection',
        'Motorboat',
        'Cargo Truck',
        'Dry Cargo Ship',
        'other-airplane',
        'Dump Truck',
        'Van',
        'Small Car']


class FAIR1M(object):
    def __init__(self, root_path):
        self.root_path = root_path 
        self.im_path = osp.join(root_path, 'images')
        self.anno_path = osp.join(root_path, 'labelXml')  
        self.im_files = glob.glob(osp.join(self.im_path,'*.tif'))
        self.anno_files = glob.glob(osp.join(self.anno_path,'*.xml'))
        self.dist_root = root_path.replace(osp.split(self.root_path)[1], osp.split(self.root_path)[1] + '_augment')
        self.dist_im_dir = osp.join(self.dist_root, 'images')
        self.dist_an_dir = osp.join(self.dist_root, 'labelXml')
        self.classes = CLASS_NAMES

        makedir(self.dist_im_dir)
        makedir(self.dist_an_dir)

    def parse_annos(self, label):
        tree = etree.parse(label)
        points = tree.xpath("//points/point/text()")
        classnames = tree.xpath("//possibleresult/name/text()")
        bboxes = np.array([eval(x) for x in points]).reshape(-1, 5, 2)[:,:-1,:].reshape(-1,8) 
        return classnames, np.array(bboxes)
    
    
    def save_ims(self, im, filename):
        dist_im = osp.join(self.dist_im_dir, filename + '.tif')
        cv2.imwrite(dist_im, im)

    def save_labels(self, classnames, bboxes, size, filename):
        node_root = Element('annotation')
        node_source = SubElement(node_root,'source')
        SubElement(node_source,'filename').text = "%s" % filename + '.tif'
        SubElement(node_source,'origin').text = 'GF2/GF3'
        node_research = SubElement(node_root,'research')
        SubElement(node_research,'version').text = '1.0'
        SubElement(node_research,'provider').text = 'FAIR1M'
        SubElement(node_research,'author').text = 'Cyber'
        SubElement(node_research,'pluginname').text = 'FAIR1M'
        SubElement(node_research,'pluginclass').text = 'object detection'
        SubElement(node_research,'time').text = '2021-07-21'
        node_size = SubElement(node_root,'size')
        SubElement(node_size,'height').text = "%s" %  size[0]
        SubElement(node_size,'width').text = "%s" %  size[1]
        SubElement(node_size,'depth').text = "%s" %  size[2]

        node_objects = SubElement(node_root, 'objects')
        for name, bbox in zip(classnames, bboxes):
                x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                node_object = SubElement(node_objects, 'object')
                SubElement(node_object,'coordinate').text = 'pixel'
                SubElement(node_object,'type').text = 'rectangle'
                SubElement(node_object,'description').text = 'None'
                node_possibleresult = SubElement(node_object,'possibleresult')
                SubElement(node_possibleresult,'name').text = "%s" % name
                
                node_points = SubElement(node_object,'points')
                SubElement(node_points,'point').text = "%s" % ','.join([format(x1,'.6f'),format(y1,'.6f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x2,'.6f'),format(y2,'.6f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x3,'.6f'),format(y3,'.6f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x4,'.6f'),format(y4,'.6f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x1,'.6f'),format(y1,'.6f')])

        xml = tostring(node_root,encoding='utf-8', method="xml",xml_declaration=True,pretty_print=True)  
        dom = parseString(xml)
        with open(osp.join(self.dist_an_dir, filename + '.xml'), 'wb') as f:
            f.write(xml)

    def save(self, im, classnames, bboxes, filename):
        self.save_ims(im, filename)
        self.save_labels(classnames, bboxes, im.shape, filename)
