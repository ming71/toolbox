
##  在root dir下直接bmp转jpg并删除原图


#############   bmp转jpg ########
# coding:utf-8
import os
from PIL import Image
from tqdm import tqdm

# bmp 转换为jpg
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

def main():
   file_path = "/data-tmp/stela-master/HRSC2016/Test/AllImages"
   bmpToJpg(file_path)
   del_bmp(file_path)

if __name__ == '__main__':
   main()

