import os
from tqdm import tqdm


root_dir = r'/data-tmp/stela-master/DOTA/split_train'

img_dir = os.path.join(root_dir, 'images')
label_dir = os.path.join(root_dir, 'labelTxt')

label_files = os.listdir(label_dir)
fsize = [os.path.getsize(os.path.join(label_dir,label_file)) for f in label_files]
for idx, label_file in tqmd(enumerate(label_files)):
    if fsize[idx] == 0:
        import ipdb; ipdb.set_trace()
        filename = os.path.splitext(label_file)[0]
        os.remove(os.path.join(img_dir, filename+'.png'))
        os.remove(os.path.join(label_dir, label_file))
    else:
        continue