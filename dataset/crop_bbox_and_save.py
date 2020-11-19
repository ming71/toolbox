import numpy as np
import cv2
import os
import ipdb



# 功能：输入图像和label的路径，根据label裁剪出gt并另存
# 注意：目标太多，最好限制输入list的图片数目！！这里用的MAX_IMAGES


# 提取label的信息，将所有目标的坐标打包输出(这里写的是xml格式，有需要再其他的自定义)
def get_points(xml_path):
    with open(xml_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<object>')	
        objects.pop(0)
        assert len(objects) > 0, 'No object found in ' + xml_path 

        object_coors=[]	# coor内一个元素是一个物体，含4坐标
        for object in objects:
             xmin = object[object.find('<xmin>')+6 : object.find('</xmin>')]
             ymin = object[object.find('<ymin>')+6 : object.find('</ymin>')]
             xmax = object[object.find('<xmax>')+6 : object.find('</xmax>')]
             ymax = object[object.find('<ymax>')+6 : object.find('</ymax>')]
             object_coors.append(np.array([int(xmin),int(ymin),int(xmax),int(ymax)]))
    return object_coors  


# 输入：单张图片和其标注xml提取的所有物体坐标的list  ,  save_flag
def crop_and_save(img_path,object_coors,save_flag=False,save_path=None):
    img=cv2.imread(img_path)
    for cnt,coor in enumerate(object_coors):
        # ipdb.set_trace()
        img_obj = img[coor[1]:coor[3],coor[0]:coor[2]]	
        if save_flag:
            save_obj_path=os.path.join(save_path,str(cnt)+'obj'+img_path.split('/')[-1])
            # print(save_obj_path)
            cv2.imwrite(save_obj_path,img_obj)
    #     cv2.imshow('crop_box',img_obj)
    # cv2.waitKey(0)

if __name__ == "__main__":
    img_path = '/py/datasets/ship/ships/image'
    xml_path = '/py/datasets/ship/ships/label'
    save_path = '/py/toolbox/GrabCut/ship'

    MAX_IMAGES = 200

    if os.path.isdir(img_path) and os.path.isdir(img_path):
        img_files = os.listdir(img_path)
        xml_files = os.listdir(xml_path)
        img_files.sort()	# 进行排序，使得xml和img序列一致可以zip
        xml_files.sort()
        iterations = zip(img_files[:MAX_IMAGES],xml_files[:MAX_IMAGES])
        for iter in iterations:
            object_coors = get_points(os.path.join(xml_path,iter[1]))
            crop_and_save(os.path.join(img_path,iter[0]),object_coors,save_flag=True,save_path=save_path)
    else:
        print('Path Not Matched!!!')
