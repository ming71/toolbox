import os
import glob
import shutil
import os.path as osp
from tqdm import tqdm
from shutil import copyfile
from IC152DOTA import generate_txt_labels
from IC152JSON import generate_json_labels

def dataset_partition(data_dir):
    train_dir = osp.join(data_dir,'Train')
    test_dir = osp.join(data_dir, 'Test')

    dirs = [train_dir, test_dir]
    datatypes = ['train', 'test']   # 注意IC15的原始默认训练和测试集名字改为'train', 'test'
    for idx, train_dir in enumerate(dirs):
        datatype = datatypes[idx]
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(osp.join(train_dir, 'images'))
        os.makedirs(osp.join(train_dir, 'labels'))
        im_files = glob.glob(osp.join(data_dir,datatype)+'/*.jpg')
        an_files = glob.glob(osp.join(data_dir,datatype)+'/*.txt')
        dst_im_files = [x.replace(datatype, datatype.capitalize()+'/images') for x in im_files]
        dst_an_files = [x.replace(datatype, datatype.capitalize()+'/labels') for x in an_files]
        trainbar = tqdm(range(len(dst_an_files)))
        for idx in trainbar:
            trainbar.set_description(datatype + " partition")
            copyfile(im_files[idx], dst_im_files[idx])
            copyfile(an_files[idx], dst_an_files[idx])


def preprare_ic15(data_dir):
    train_dir = osp.join(data_dir,'Train')
    test_dir = osp.join(data_dir, 'Test')
    # convert ucas_aod to dota raw format
    generate_txt_labels(train_dir)
    generate_txt_labels(test_dir)
    # convert it to json format
    generate_json_labels(train_dir,osp.join(data_dir,'train.json'))
    generate_json_labels(test_dir,osp.join(data_dir,'test.json'))

if __name__ == '__main__':
    IC15_dir = '/data-input/RotationDet/data/ICDAR2015'
    dataset_partition(IC15_dir)
    preprare_ic15(IC15_dir)
    print('done')