import os 
import glob
import random
import shutil

from tqdm import tqdm

random.seed(666)

def copyfiles(src_files, dst_folder):
    pbar = tqdm(src_files)
    for file in pbar:
        filename = os.path.split(file)[1]
        pbar.set_description("Creating {}:".format(dst_folder))
        dstfile = os.path.join(dst_folder, filename)
        # print(dstfile)
        shutil.copyfile(file, dstfile)




def creat_tree(root_dir):
    if not os.path.exists(root_dir):
        raise RuntimeError('invalid dataset path!')
    os.mkdir(os.path.join(root_dir, 'AllImages'))
    os.mkdir(os.path.join(root_dir, 'Annotations'))
    os.mkdir(os.path.join(root_dir, 'ImageSets'))
    imgs = glob.glob(os.path.join(root_dir, 'positive image set/*.jpg'))
    annos = glob.glob(os.path.join(root_dir, 'ground truth/*.txt'))
    copyfiles(imgs, os.path.join(root_dir, 'AllImages') ) 
    copyfiles(annos, os.path.join(root_dir, 'Annotations') ) 


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
            src = os.path.join(img_dir, filename + '.jpg')
            dst = os.path.join(test_dir, filename + '.jpg')
            shutil.copyfile(src, dst)

def generate_imageset_file(root_dir):
    split_ratio=[0.8, 0.2]
    train_r, test_r = split_ratio
    data = set([x for x in range(1,651)])
    uniform = True
    if uniform:
        ratio = 5
        testset = set(list(data)[::5])
    else:
        testset = set(random.sample(data,int(650 * test_r)))
    trainset = data - testset  
    with open(os.path.join(root_dir, 'ImageSets/train.txt'), 'w') as f1:
        s1 = '\n'.join([ str(x).zfill(3) for x in trainset])
        f1.write(s1)
    with open(os.path.join(root_dir, 'ImageSets/test.txt'), 'w') as f2:
        s2 = '\n'.join([ (str(x)).zfill(3) for x in testset])
        f2.write(s2)


if __name__ == "__main__":
    root_dir = '/py/datasets/NWPU VHR-10 dataset'
    creat_tree(root_dir)
    generate_imageset_file(root_dir)
    generate_test(root_dir)
