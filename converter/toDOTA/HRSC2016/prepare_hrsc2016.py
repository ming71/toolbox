import os
import shutil
import os.path as osp
from tqdm import tqdm
from shutil import copyfile
from HRSC2DOTA import generate_txt_labels
from HRSC2JSON import generate_json_labels

def dataset_partition(data_dir):
    train_dir = osp.join(data_dir,'Train')
    val_dir = osp.join(data_dir,'Val')
    trainval_dir = osp.join(data_dir,'Trainval')
    test_dir = osp.join(data_dir, 'Test')
    imgsets = osp.join(data_dir, 'ImageSets')
    trainset = osp.join(imgsets, 'train.txt')
    valset = osp.join(imgsets, 'val.txt')
    trainvelset = osp.join(imgsets, 'trainval.txt')
    testset = osp.join(imgsets, 'test.txt')

    train_dirs = [train_dir, val_dir, trainval_dir, test_dir]
    trainsets = [trainset, valset, trainvelset, testset]
    datatypes = ['train', 'val', 'trainval', 'test']
    for idx, train_dir in enumerate(train_dirs):
        datatype = datatypes[idx]
        trainset = trainsets[idx]
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(osp.join(train_dir, 'images'))
        os.makedirs(osp.join(train_dir, 'annotations'))
        with open(trainset, 'r') as f1:
            train_content = f1.readlines()
        trainbar = tqdm(train_content)
        for trf in trainbar:
            trainbar.set_description(datatype + " partition")
            src_im = osp.join(data_dir + '/FullDataSet/AllImages', trf.strip() + '.jpg')
            dst_im = osp.join(train_dir + '/images', trf.strip() + '.jpg')
            src_an = osp.join(data_dir + '/FullDataSet/Annotations', trf.strip() + '.xml')
            dst_an = osp.join(train_dir + '/annotations', trf.strip() + '.xml')
            copyfile(src_im, dst_im)
            copyfile(src_an, dst_an)


def preprare_hrsc2016(data_dir):
    train_dir = osp.join(data_dir,'Train')
    val_dir = osp.join(data_dir,'Val')
    trainval_dir = osp.join(data_dir,'Trainval')
    test_dir = osp.join(data_dir, 'Test')
    imgset_dir = osp.join(data_dir, 'ImageSets')
    # convert hrsc2016 to dota raw format
    generate_txt_labels(train_dir)
    generate_txt_labels(val_dir)
    generate_txt_labels(trainval_dir)
    generate_txt_labels(test_dir)
    # convert it to json format
    generate_json_labels(train_dir,osp.join(imgset_dir,'train.json'))
    generate_json_labels(val_dir,osp.join(imgset_dir,'val.json'))
    generate_json_labels(trainval_dir,osp.join(imgset_dir,'trainval.json'))
    generate_json_labels(test_dir,osp.join(imgset_dir,'test.json'))

if __name__ == '__main__':
    hrsc2016_dir = '/data-input/RotationDet/data/HRSC2016'
    dataset_partition(hrsc2016_dir)
    preprare_hrsc2016(hrsc2016_dir)
    print('done')