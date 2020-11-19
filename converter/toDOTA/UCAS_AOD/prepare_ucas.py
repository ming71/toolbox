import os
import shutil
import os.path as osp
from tqdm import tqdm
from shutil import copyfile
from UCAS2DOTA import generate_txt_labels
from UCAS2JSON import generate_json_labels

def dataset_partition(data_dir):
    train_dir = osp.join(data_dir,'Train')
    val_dir = osp.join(data_dir,'Val')
    test_dir = osp.join(data_dir, 'Test')
    imgsets = osp.join(data_dir, 'ImageSets')
    trainset = osp.join(imgsets, 'train.txt')
    valset = osp.join(imgsets, 'val.txt')
    testset = osp.join(imgsets, 'test.txt')

    train_dirs = [train_dir, val_dir, test_dir]
    trainsets = [trainset, valset, testset]
    datatypes = ['train', 'val', 'test']
    for idx, train_dir in enumerate(train_dirs):
        datatype = datatypes[idx]
        trainset = trainsets[idx]
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(osp.join(train_dir, 'images'))
        os.makedirs(osp.join(train_dir, 'labels'))
        with open(trainset, 'r') as f1:
            train_content = f1.readlines()
        trainbar = tqdm(train_content)
        for trf in trainbar:
            trainbar.set_description(datatype + " partition")
            src_im = osp.join(data_dir + '/AllImages', trf.strip() + '.png')
            dst_im = osp.join(train_dir + '/images', trf.strip() + '.png')
            src_an = osp.join(data_dir + '/Annotations', trf.strip() + '.txt')
            dst_an = osp.join(train_dir + '/annotations', trf.strip() + '.txt')
            copyfile(src_im, dst_im)
            copyfile(src_an, dst_an)


def preprare_ucas_aod(data_dir):
    train_dir = osp.join(data_dir,'Train')
    val_dir = osp.join(data_dir,'Val')
    test_dir = osp.join(data_dir, 'Test')
    imgset_dir = osp.join(data_dir, 'ImageSets')
    # convert ucas_aod to dota raw format
    # generate_txt_labels(train_dir)
    # generate_txt_labels(val_dir)
    # generate_txt_labels(test_dir)
    # convert it to json format
    generate_json_labels(train_dir,osp.join(imgset_dir,'train.json'))
    generate_json_labels(val_dir,osp.join(imgset_dir,'val.json'))
    generate_json_labels(test_dir,osp.join(imgset_dir,'test.json'))

if __name__ == '__main__':
    ucas_aod_dir = '/data-input/RotationDet/data/UCAS_AOD'
    dataset_partition(ucas_aod_dir)
    preprare_ucas_aod(ucas_aod_dir)
    print('done')