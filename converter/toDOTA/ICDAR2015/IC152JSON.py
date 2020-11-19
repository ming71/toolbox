import os
import sys
import json
import xmltodict
from PIL import Image
import os.path as osp
from tqdm import tqdm

sys.path.append("..")
from dota_poly2rbox import poly2rbox_single_v2


def parse_ann_info(img_base_path, label_base_path, img_name):
    lab_path = osp.join(label_base_path, img_name+'.txt')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            bbox = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = poly2rbox_single_v2(bbox)
            class_name = ann_line[8]
            difficult = int(ann_line[9])
            # ignore difficult =2
            if difficult == 0:
                bboxes.append(bbox)
                labels.append(class_name)
            elif difficult == 1:
                bboxes_ignore.append(bbox)
                labels_ignore.append(class_name)
    return bboxes, labels, bboxes_ignore, labels_ignore


def generate_json_labels(src_path, out_path, trainval=True):
    """Generate .json labels which is similar to coco format
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output json file path
        trainval: trainval or test?
    """
    img_path = osp.join(src_path, 'images') if trainval == True else src_path
    label_path = os.path.join(src_path, 'labelTxt')
    img_lists = os.listdir(img_path)

    data_dict = []

    with open(out_path, 'w') as f:
        pbar = tqdm(img_lists)
        for id, img in enumerate(pbar):
            pbar.set_description("ICDAR2015 generate_json_labels in {}".format(src_path))
            img_info = {}
            img_name = osp.splitext(img)[0]
            label = os.path.join(label_path, img_name+'.txt')
            img = Image.open(osp.join(img_path, img))
            img_info['filename'] = img_name + '.jpg' 
            img_info['height'] = img.height
            img_info['width'] = img.width
            img_info['id'] = id
            if(trainval == True):
                if(os.path.exists(label) == False):
                    print('Label:'+img_name+'.txt'+' Not Exist')
                else:
                    bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(
                        img_path, label_path, img_name)
                    ann = {}
                    ann['bboxes'] = bboxes
                    ann['labels'] = labels
                    ann['bboxes_ignore'] = bboxes_ignore
                    ann['labels_ignore'] = labels_ignore
                    img_info['annotations'] = ann
            data_dict.append(img_info)
        json.dump(data_dict, f)



if __name__ == '__main__':
    generate_json_labels('/data-input/RotationDet/data/ICDAR2015/airplane/train',
                         '/data-input/RotationDet/data/ICDAR2015/airplane/train/train.json')
    generate_json_labels('/data-input/RotationDet/data/ICDAR2015/airplane/val',
                         '/data-input/RotationDet/data/ICDAR2015/airplane/val/test.json', trainval=False)
    print('done!')
