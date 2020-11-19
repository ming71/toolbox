import os
import sys
import json
import shutil
import zipfile
import argparse
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config
from tqdm import tqdm

from mmdet.datasets import build_dataset
from mmdet.core.bbox import rbox2poly_single

from lxml.etree import Element,SubElement,tostring
import xml.etree.ElementTree as ElementTree
from xml.dom.minidom import parseString
import xml.dom.minidom


def creat_submission(data_test, dets, dstpath, classnames):
    provider = 'Beijing Institute of Technology'
    author = r'理工大附小'
    pluginname = 'Airplane Detection and Recognition in Optical Images'

    det_folder = osp.join(dstpath, 'test')
    if osp.exists(det_folder):
        shutil.rmtree(det_folder)
    os.mkdir(det_folder)

    with open(data_test.ann_file,'r') as ft:
        images = json.load(ft)
        im_names = [osp.splitext(x['filename'])[0] for x in images] 

    pbar = tqdm(dets)
    for img_di, det in enumerate(pbar):
        pbar.set_description("creat_submission")
        im_name = im_names[img_di]
        # if im_name != '5':
        #     continue
        node_root=Element('annotation')
        node_source = SubElement(node_root,'source')
        SubElement(node_source,'filename').text = "%s" % im_name + '.tif'
        SubElement(node_source,'origin').text = 'GF2/GF3'
        node_research = SubElement(node_root,'research')
        SubElement(node_research,'version').text = '4.0'
        SubElement(node_research,'provider').text = "%s" % provider
        SubElement(node_research,'author').text = "%s" % author
        SubElement(node_research,'pluginname').text = "%s" % pluginname
        SubElement(node_research,'pluginclass').text = 'Detection'
        SubElement(node_research,'time').text = '2020-07-2020-11'
        nt = sum([len(x) for x in det])
        if nt == 0:
            det[0]=np.array([[0 for i in range(9)]]).astype(np.float32)
        node_objects = SubElement(node_root, 'objects')
        for cls_id, cls_dets in enumerate(det):
            # import ipdb;ipdb.set_trace()
            for obj in cls_dets:
                conf = obj[-1]
                x1, y1, x2, y2, x3, y3, x4, y4 = rbox2poly_single(obj[:-1])
                node_object = SubElement(node_objects, 'object')
                SubElement(node_object,'coordinate').text = 'pixel'
                SubElement(node_object,'type').text = 'rectangle'
                SubElement(node_object,'description').text = 'None'
                node_possibleresult = SubElement(node_object,'possibleresult')
                SubElement(node_possibleresult,'name').text = "%s" % classnames[cls_id]
                SubElement(node_possibleresult,'probability').text = "%s" % format(conf,'.3f')
                node_points = SubElement(node_object,'points')
                SubElement(node_points,'point').text = "%s" % ','.join([format(x1,'.1f'),format(y1,'.1f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x2,'.1f'),format(y2,'.1f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x3,'.1f'),format(y3,'.1f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x4,'.1f'),format(y4,'.1f')])
                SubElement(node_points,'point').text = "%s" % ','.join([format(x1,'.1f'),format(y1,'.1f')])

        xml = tostring(node_root,encoding='utf-8', method="xml",xml_declaration=True,pretty_print=True)  
        dom = parseString(xml)
        with open(osp.join(det_folder, im_name + '.xml'), 'wb') as f:
            f.write(xml)
    os.system('cd {} && zip -r -q  {} {} '.format(dstpath, 'test.zip', 'test'))
    shutil.rmtree(det_folder)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', default='configs/DOTA/faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5.py')
    args = parser.parse_args()
    return args

def parse_results(config_file, resultfile, dstpath ):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data['test']
    dataset = build_dataset(data_test)
    outputs = mmcv.load(resultfile)
    dataset_type = cfg.dataset_type
    classnames = dataset.CLASSES
    creat_submission(data_test, outputs, dstpath, classnames)



if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    config_name = osp.splitext(osp.basename(config_file))[0]
    pkl_file = osp.join('work_dirs', config_name, 'results.pkl')
    output_path = osp.join('work_dirs', config_name)
    parse_results(config_file, pkl_file, output_path)

