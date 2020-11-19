import os 
import glob
import random
import shutil

from tqdm import tqdm

random.seed(666)

def copyfiles(src_files, dst_folder, is_plane = False):
    pbar = tqdm(src_files)
    for file in pbar:
        pbar.set_description("Creating {}:".format(dst_folder))
        if not is_plane:
            filename = os.path.split(file)[1]
        else: 
            _filename = os.path.split(file)[1]
            name, ext = os.path.splitext(_filename)
            filename = 'P' + str(int(name.strip('P')) + 510).zfill(4) + ext
        dstfile = os.path.join(dst_folder, filename)
        # print(dstfile)
        shutil.copyfile(file, dstfile)


def rewrite_label(annos, dst_folder, is_plane = False):
    pbar = tqdm(annos)
    for file in pbar:
        pbar.set_description("Rewriting to {}:".format(dst_folder))
        if not is_plane:
            filename = os.path.split(file)[1]
        else: 
            _filename = os.path.split(file)[1]
            name, ext = os.path.splitext(_filename)
            filename = 'P' + str(int(name.strip('P')) + 510).zfill(4) + ext
        dstfile = os.path.join(dst_folder, filename)
        # print(dstfile)
        with open(dstfile, 'w') as fw:
            with open(file, 'r') as f:
                _lines = f.readlines()
                if is_plane:
                    lines = ['airplane  ' + x for x in _lines]  
                else:
                    lines = ['car  ' + x for x in _lines]  
                content = ''.join(lines)
                fw.write(content)

def creat_tree(root_dir):
    if not os.path.exists(root_dir):
        raise RuntimeError('invalid dataset path!')
    os.mkdir(os.path.join(root_dir, 'AllImages'))
    os.mkdir(os.path.join(root_dir, 'Annotations'))
    car_imgs = glob.glob(os.path.join(root_dir, 'CAR/*.png'))
    car_annos = glob.glob(os.path.join(root_dir, 'CAR/P*.txt'))
    airplane_imgs = glob.glob(os.path.join(root_dir, 'PLANE/*.png'))
    airplane_annos = glob.glob(os.path.join(root_dir, 'PLANE/P*.txt'))   
    copyfiles(car_imgs,  os.path.join(root_dir, 'AllImages') ) 
    copyfiles(airplane_imgs,  os.path.join(root_dir, 'AllImages'), True)
    rewrite_label(car_annos, os.path.join(root_dir, 'Annotations'))
    rewrite_label(airplane_annos, os.path.join(root_dir, 'Annotations'), True)


def generate_test(root_dir):
    setfile = os.path.join(root_dir, 'ImageSets/test.txt')
    img_dir = os.path.join(root_dir, 'AllImages')
    test_dir = os.path.join(root_dir, 'Test')
    os.makedirs(test_dir)
    if not os.path.exists(setfile):
        raise RuntimeError('{} is not founded!'.format(setfile))
    with open(setfile, 'r') as f:
        lines = f.readlines()
        pbar = tqdm(lines)
        for line in pbar:
            pbar.set_description("Copying to Test dir...")
            filename = line.strip()
            src = os.path.join(img_dir, filename + '.png')
            dst = os.path.join(test_dir, filename + '.png')
            shutil.copyfile(src, dst)

# DON'T run this function, casuse you cannot make sure the same result will be generated.
# def generate_imageset_file(root_dir):
#     split_ratio=[0.5, 0.2, 0.3]
#     train_r, val_r, test_r = split_ratio
#     car = set([x for x in range(1,511)])
#     airplane = set([x for x in range(511, 1511)])
#     train_car = set(random.sample(car,int(510 * train_r)))
#     val_car = set(random.sample(car - train_car,int(510 * val_r)))
#     test_car = car - val_car - train_car
#     train_airplane = set(random.sample(airplane,int(1000 * train_r)))
#     val_airplane = set(random.sample(airplane - train_airplane,int(1000 * val_r)))
#     test_airplane = airplane - val_airplane - train_airplane
#     train_idx = train_car | train_airplane
#     val_idx = val_car | val_airplane
#     test_idx = test_car | test_airplane
#     with open(os.path.join(root_dir, 'ImageSets/train.txt'), 'w') as f1:
#         s1 = '\n'.join(['P' + str(x).zfill(4) for x in train_idx])
#         f1.write(s1)
#     with open(os.path.join(root_dir, 'ImageSets/val.txt'), 'w') as f2:
#         s2 = '\n'.join(['P' + (str(x)).zfill(4) for x in val_idx])
#         f2.write(s2)
#     with open(os.path.join(root_dir, 'ImageSets/test.txt'), 'w') as f3:
#         s3 = '\n'.join(['P' + str(x).zfill(4) for x in test_idx])
#         f3.write(s3)


if __name__ == "__main__":
    root_dir = '/data-input/RotationDet/data/UCAS_AOD'
    creat_tree(root_dir)
    generate_test(root_dir)
