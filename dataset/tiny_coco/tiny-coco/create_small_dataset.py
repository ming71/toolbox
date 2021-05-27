import json,os
import pdb, time
from pycocotools.coco import COCO
import shutil

####################################################################################################
source_root = "/home/ubuntu/mscoco"
datadir = os.path.join(source_root, 'annotations')
json_file = "instances_train2017.json"

dest_root = "small_coco"
new_json_file = "instances_train2017_small.json"
new_imgs_name = "train_2017_small" 

#regenerate = True 
regenerate = False 
num = 64
images_list = [391895, 522418, 184613, 318219, 554625, 574769, 60623, 309022, 5802, 222564, 118113, 193271, 224736, 483108, 403013, 374628]
#images_list = [391895]
####################################################################################################


def generate_annotations(images_list=images_list):
    with open(os.path.join(datadir, json_file), 'r') as f:  
        coco = json.load(f)

    new_images = []
    new_annotations = []

    # look for images
    for img in coco['images']:
        if img['id'] in images_list:
            new_images.append(img)

    # look for annotations
    for ann in coco['annotations']:
        if ann['image_id'] in images_list:
            new_annotations.append(ann)

    # update and save
    coco['images'] = new_images
    coco['annotations'] = new_annotations

    print("begin to save")
    with open( os.path.join(dest_root, new_json_file), 'w') as ff:
        json.dump(coco, ff)


def generate_images(images_list=images_list):
    new_dir_path = os.path.join(dest_root, new_imgs_name)  

    if os.path.exists(new_dir_path):
        shutil.rmtree(new_dir_path)
    os.mkdir(new_dir_path) 

    for img in images_list:
        img_name = format(img, "012") + ".jpg"
        src_file  = os.path.join(source_root, "train2017", img_name)
        dest_file = os.path.join(new_dir_path, img_name)
        shutil.copyfile(src_file, dest_file) 
    


if regenerate:
    images_list = [item['id'] for item in coco['images'][:num]]
    print(images_list)

if os.path.exists(dest_root):
    shutil.rmtree(dest_root)
os.mkdir(dest_root)

generate_annotations(images_list)
generate_images(images_list)


