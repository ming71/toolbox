import os
import glob
import shutil
import os.path as osp
from tqdm import tqdm
from shutil import copyfile
from MSRA_TD5002DOTA import generate_txt_labels
from MSRA_TD5002JSON import generate_json_labels

def dataset_partition(data_dir):
    train_dir = osp.join(data_dir,'Train')
    test_dir = osp.join(data_dir, 'Test')

    dirs = [train_dir, test_dir]
    datatypes = ['train', 'test']   
    for idx, train_dir in enumerate(dirs):
        datatype = datatypes[idx]
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(osp.join(train_dir, 'images'))
        os.makedirs(osp.join(train_dir, 'labels'))
        im_files = glob.glob(osp.join(data_dir,datatype)+'/*.JPG')
        an_files = glob.glob(osp.join(data_dir,datatype)+'/*.gt')
        dst_im_files = [x.replace(datatype, datatype.capitalize()+'/images').replace('JPG', 'jpg') for x in im_files]
        dst_an_files = [x.replace(datatype, datatype.capitalize()+'/labels').replace('gt', 'txt') for x in an_files]
        trainbar = tqdm(range(len(dst_an_files)))
        for idx in trainbar:
            trainbar.set_description(datatype + " partition")
            copyfile(im_files[idx], dst_im_files[idx])
            copyfile(an_files[idx], dst_an_files[idx])


def preprare_MSRA_TD500(data_dir):
    train_dir = osp.join(data_dir,'Train')
    test_dir = osp.join(data_dir, 'Test')
    # convert to dota txt format
    generate_txt_labels(train_dir)
    generate_txt_labels(test_dir)
    # convert it to json format
    generate_json_labels(train_dir,osp.join(data_dir,'train.json'))
    generate_json_labels(test_dir,osp.join(data_dir,'test.json'))

if __name__ == '__main__':
    root_dir = '/data-input/RotationDet/data/MSRA_TD500'
    dataset_partition(root_dir)
    preprare_MSRA_TD500(root_dir)
    print('done')