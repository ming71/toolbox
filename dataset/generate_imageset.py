import os
import sys
import glob
from PIL import Image
from tqdm import tqdm

DATASETS = ['IC15', 'IC13',
            'HRSC2016', 'DOTA', 'UCAS_AOD', 'NWPU_VHR' ,
            'GaoFenShip', 'GaoFenAirplane', 
            'VOC']


def bmpToJpg(file_path):
   for fileName in tqdm(os.listdir(file_path)):
       newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
       im = Image.open(os.path.join(file_path,fileName))
       rgb = im.convert('RGB')      #灰度转RGB
       rgb.save(os.path.join(file_path,newFileName))

def del_bmp(root_dir=None):
    file_list = os.listdir(root_dir)
    for f in file_list:
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            if f.endswith(".BMP") or f.endswith(".bmp"):
                os.remove(file_path)
                print( " File removed! " + file_path)
        elif os.path.isdir(file_path):
            del_bmp(file_path)




def generate_iamgets(dataset=None):
    assert dataset in DATASETS, 'Not supported dataset'
    if dataset  == 'DOTA':
        # For DOTA
        # train_img_path = r"/data-input/das_dota/DOTA/trainsplit/images" 
        # val_img_path = r"/data-input/das_dota/DOTA/valsplit/images" 
        # set_file = r'/data-input/das_dota/DOTA/trainval.txt'
        files= sorted(glob.glob(os.path.join(train_img_path, '**.*' ))) + sorted(glob.glob(os.path.join(val_img_path, '**.*' )))
        with open(set_file,'w') as f:
            for file in files:
                img_path, filename = os.path.split(file)
                name, extension = os.path.splitext(filename)
                if extension in ['.jpg', '.bmp','.png']:
                    f.write(os.path.join(file)+'\n')

    # image和label分开或者丢一块都成
    elif dataset in ['IC13', 'GaoFenShip']:
        # For IC13
        # train_img_dir = r"/data-input/das_dota/ICDAR13/train/images" 
        # val_img_dir = r"/data-input/das_dota/ICDAR13/val/images" 
        # trainset = r'/data-input/das_dota/ICDAR13/train.txt'
        # valset = r'/data-input/das_dota/ICDAR13/test.txt'
        # For GaoFenShip
        train_img_dir = r"/data-input/AerialDetection-master/data/ship/train" 
        val_img_dir = r"/data-input/AerialDetection-master/data/ship/val" 
        trainset = r'/data-input/AerialDetection-master/data/ship/train.txt'
        valset = r'/data-input/AerialDetection-master/data/ship/test.txt'

        for set_file, im_dir in zip([trainset, valset], [train_img_dir, val_img_dir]):
            with open(set_file,'w') as f:
                if dataset in ['IC13', 'IC15']:
                    files = glob.glob(os.path.join(im_dir, '**.jpg*' ))
                elif dataset == 'GaoFenShip':
                    files = glob.glob(os.path.join(im_dir, '**.tiff*' ))
                else:
                    raise NotImplementedError
                for file in files:
                    f.write(file+'\n')
    

    
    # voc格式的有imageset file
    elif  dataset in ['HRSC2016', 'UCAS_AOD', 'VOC', 'NWPU_VHR']:
        trainset = r'/data-input/das_dota/UCAS_AOD/ImageSets/train.txt'
        valset   = r'/data-input/das_dota/UCAS_AOD/ImageSets/test.txt'
        testset   = r'/data-input/das_dota/UCAS_AOD/ImageSets/test.txt'
        img_dir = r'/data-input/das_dota/UCAS_AOD/AllImages'
        label_dir = r'/data-input/das_dota/UCAS_AOD/Annotations'
        root_dir = r'/data-input/das_dota/UCAS_AOD' 

        for dataset in [trainset, valset, testset]:
            with open(dataset,'r') as f:
                names = f.readlines()
                if DATASET in ['HRSC2016', 'NWPU_VHR']:
                    paths = [os.path.join(img_dir,x.strip()+'.jpg\n') for x in names]
                elif DATASET == 'UCAS_AOD':
                    paths = [os.path.join(img_dir,x.strip()+'.png\n') for x in names]
                with open(os.path.join(root_dir,os.path.split(dataset)[1]), 'w') as fw:
                    fw.write(''.join(paths))


if __name__ == '__main__':
    DATASET = 'UCAS_AOD'
    generate_iamgets(DATASET)



