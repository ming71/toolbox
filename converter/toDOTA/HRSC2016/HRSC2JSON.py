import json
import os
import xmltodict
import os.path as osp
from tqdm import tqdm


def parse_ann_info(objects):
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        if obj['difficult'] == '0':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = 'ship'
            bboxes.append(bbox)
            labels.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore


def generate_json_labels(root_path, txt_ann_path, trainval=True):
    img_path = osp.join(root_path, 'images')
    label_path = osp.join(root_path, 'annotations')
    json_ann_path = osp.splitext(txt_ann_path)[
        0]+'.json'  # yourdir/trainval.json
    ims = os.listdir(img_path)    
    data_dicts = []
    pbar = tqdm(ims)
    for im in pbar:
        pbar.set_description("HRSC2016 Preparation...")
        label = osp.join(label_path, im.replace('.jpg','.xml'))
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['HRSC_Image']
        f_label.close()
        img_info = dict(
            filename=im,
            height=int(data_dict['Img_SizeHeight']),
            width=int(data_dict['Img_SizeWidth']),
            id=im,
            annotations=dict(
                bboxes=[],
                labels=[],
                bboxes_ignore=[],
                labels_ignore=[]))
        # with annotations
        if data_dict['HRSC_Objects']:
            objects = data_dict['HRSC_Objects']['HRSC_Object']
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(
                objects)
            ann = dict(
                bboxes=bboxes,
                labels=labels,
                bboxes_ignore=bboxes_ignore,
                labels_ignore=labels_ignore)
            img_info['annotations'] = ann
        data_dicts.append(img_info)

    with open(json_ann_path, 'w') as f:
        json.dump(data_dicts, f)


if __name__ == '__main__':
    generate_json_labels('/data-input/RotationDet/data/HRSC2016',
                         "/data-input/RotationDet/data/HRSC2016/ImageSets/trainval.txt")
    generate_json_labels('/data-input/RotationDet/data/HRSC2016',
                         "/data-input/RotationDet/data/HRSC2016/ImageSets/test.txt")
    print('done!')
