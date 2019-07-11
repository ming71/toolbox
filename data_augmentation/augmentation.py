
# 一个问题：这里的变换是直接生成新的文件夹无论原图变换与否都会重新生成的
# 如果数据集比较大，可以将src与dst写成一个，这样就能覆盖原图节省内存，但是慎重，确保无误再这么做！
# 所有变换均为随机的，如果想有目的地进行增强指定部分数据，去掉random的部分即可


import cv2
import ipdb
import os
import numpy as np
import random
import math

# ipdb.set_trace()


# 仿射变换：旋转/缩放/平移/裁剪
#Translation    +/- 10% (vertical and horizontal)
#Rotation   +/- 15 degrees
#Shear  +/- 2 degrees (vertical and horizontal)
#Scale  +/- 10%

def Affine(img,img_file,img_dst,xml_src,xml_dst):
    # -------变换选择-------
    Shear=True
    RotationScale=True
    Translation=True
    # -----变换参数设置------
    degrees=(-15, 15)
    translate=(.1, .1)
    scale=(.9, 1.1)
    shear=(-2, 2)
    borderValue=(127.5, 127.5, 127.5)   # 超出边界用灰色填充

    height = img.shape[1]
    width  = img.shape[0]

    if  RotationScale:
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]  # 在给定角度范围内生成一个随机旋转角
        s = random.random() * (scale[1] - scale[0]) + scale[0]  # 随机生成范围内的缩放比
        # 旋转变换矩阵，分别输入旋转中心/旋转角/缩放比(旋转角为角度制；旋转中心就是图片中心像素坐标；缩放比)
        # R阵最后一行 0 0 1非2D变换的参数信息 舍弃，得到R[:2]为旋转变化阵
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    if Translation:
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0]   # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1]   # y translation (pixels)

    if Shear:
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  #  ORDER IS IMPORTANT HERE!!

    imw = cv2.warpPerspective(img, M, dsize=(height, width), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    cv2.imwrite(os.path.join(img_dst,img_file), imw)
    rewrite_label('affine',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml',M)


# 0.5概率完成旋转和缩放
def Rot_Scale(img,img_file,img_dst,xml_src,xml_dst):
    a=0
    s=1
    if random.random() > 0 :
        scale = (0.5, 2.0)
        degrees = (0, 0)
        center=None
        if center is None:  # 不建议改，如果改，需要重写一下rewrite_label部分
            center = (img.shape[0] // 2, img.shape[1] // 2) 
        s = random.random() * (scale[1] - scale[0]) + scale[0]      # 生成随机角度和缩放比
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        M = cv2.getRotationMatrix2D(center, a, s)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite(os.path.join(img_dst,img_file), img)
    rewrite_label('rot_scale',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml',M)




# 0.5概率边缘模糊
def Blur(img,img_file,img_dst,xml_src,xml_dst):
    if  random.random() > 0.5 :
        img=cv2.GaussianBlur(img,(7,7),7)
    cv2.imwrite(os.path.join(img_dst,img_file), img)
    rewrite_label('blur',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml')



# 功能：以一定概率加特定模式的噪声(暂时只有高斯)
def Noise(img,img_file,img_dst,xml_src,xml_dst,mode='Gausssian'):
    if mode == 'Gausssian' and random.random() > 0.5 :
        noise = np.random.normal(0, 20, img.shape)
        img = img + noise.astype(int)    # 高斯噪声转为int型便于相加
        np.maximum(img,0)   # 规范到0-255内
        np.minimum(img,255)
    elif mode == 'Impulse' and random.random() > 0.1 :
        pass
    else:
        pass
    cv2.imwrite(os.path.join(img_dst,img_file), img)
    rewrite_label('noise',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml')


# 功能：50%几率随机上下或左右翻转
# 输入：像素矩阵，图像名称，目标图像文件夹路径，源xml文件夹路径，目标xml文件夹路径
# 注意：只开一个（上下/左右），否则路径报错（懒得写）
def Flip(img,img_file,img_dst,xml_src,xml_dst):
    lr_flip = True      # random left-right flip
    ud_flip = False     # random up-down flip

    if lr_flip and random.random() > 0.5:
        img = np.fliplr(img)
        rewrite_label('lr_flip',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml')
    elif ud_flip and random.random() > 0.5:
        img = np.flipud(img)
        rewrite_label('ud_flip',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml')
    else:
        rewrite_label('hsv',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml')

    cv2.imwrite(os.path.join(img_dst,img_file), img)




#  功能：随机HSV色彩空间变换：饱和度和明度提高50%；并输出label和图像文件
#  输入：像素矩阵，单个图像名称，目标图像文件夹路径，源xml文件夹路径，目标xml文件夹路径
def Hsv(img,img_file,img_dst,xml_src,xml_dst):
    fraction = 0.50
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #色彩空间变换
    #类似于BGR，HSV的shape=(w,h,c)，其中三通道的c[0,1,2]含有h,s,v信息
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

    cv2.imwrite(os.path.join(img_dst,img_file), img)
    rewrite_label('hsv',xml_src+'/'+img_file[:-4]+'.xml',xml_dst+'/'+img_file[:-4]+'.xml')


# 重写label的xml文件
# 输入输出均是是单个文件名的绝对路径
def rewrite_label(flag,xml_src_,xml_dst_,M = None):
    if flag in ['hsv','noise','blur'] :
        os.system('cp -r {} {}'.format(xml_src_,xml_dst_))


    if flag == 'lr_flip':
        with open(xml_dst_,'w') as fw:
            w,_,coor = read_xml(xml_src_)
            with open(xml_src_,'r') as fr:        
                contents=fr.read()
                objects=contents.split('<object>')
                _contents=objects.pop(0)
                for i,object in enumerate(objects): # 物体数和coor的坐标对数匹配
                    object = object[ : object.find('<x0>')+4] + str(w-coor[i][0]) + object[object.find('</x0>') :]
                    object = object[ : object.find('<x1>')+4] + str(w-coor[i][2]) + object[object.find('</x1>') :]
                    object = object[ : object.find('<x2>')+4] + str(w-coor[i][4]) + object[object.find('</x2>') :]
                    object = object[ : object.find('<x3>')+4] + str(w-coor[i][6]) + object[object.find('</x3>') :]
                    _contents=_contents + '<object>' + object
                fw.write(_contents)


    if flag == 'ud_flip':
        with open(xml_dst_,'w') as fw:
            _,h,coor = read_xml(xml_src_)
            with open(xml_src_,'r') as fr:        
                contents=fr.read()
                objects=contents.split('<object>')
                _contents=objects.pop(0)
                for i,object in enumerate(objects): 
                    object = object[ : object.find('<y0>')+4] + str(h-coor[i][1]) + object[object.find('</y0>') :]
                    object = object[ : object.find('<y1>')+4] + str(h-coor[i][3]) + object[object.find('</y1>') :]
                    object = object[ : object.find('<y2>')+4] + str(h-coor[i][5]) + object[object.find('</y2>') :]
                    object = object[ : object.find('<y3>')+4] + str(h-coor[i][7]) + object[object.find('</y3>') :]
                    _contents=_contents + '<object>' + object
                fw.write(_contents)
        

    if flag in ['rot_scale','affine'] :
        with open(xml_dst_,'w') as fw:
            w,h,coors = read_xml(xml_src_)
            rotated_coors = []   # 每个元素一个物体，内嵌四对处理后的xy tuple坐标
            for coor in coors:
                xs = coor[::2]  # 切片索引出x ，如array([931, 950, 980, 959])
                ys = coor[1::2]
                rotated_coor=[] # 存放当前物体旋转后的zip坐标，如[(931, 646), (950, 633), (980, 680), (959, 692)]
                for xy in  zip(xs,ys)  :
                    # ipdb.set_trace()
                    rotx,roty = cal_rot_box(xy[0],xy[1],M)
                    rotated_coor.append( (int(rotx),int(roty)) )

                rotated_coors.append(rotated_coor) # 逐个物体地添加

            with open(xml_src_,'r') as fr:        
                contents=fr.read()
                objects=contents.split('<object>')
                _contents=objects.pop(0)
                for i,object in enumerate(objects):
                    # print(rotated_coors[i]) 
                    # ipdb.set_trace() 
                    object = object[ : object.find('<x0>')+4] + str(rotated_coors[i][0][0]) + object[object.find('</x0>') :]
                    object = object[ : object.find('<y0>')+4] + str(rotated_coors[i][0][1]) + object[object.find('</y0>') :]
                    object = object[ : object.find('<x1>')+4] + str(rotated_coors[i][1][0]) + object[object.find('</x1>') :]
                    object = object[ : object.find('<y1>')+4] + str(rotated_coors[i][1][1]) + object[object.find('</y1>') :]
                    object = object[ : object.find('<x2>')+4] + str(rotated_coors[i][2][0]) + object[object.find('</x2>') :]
                    object = object[ : object.find('<y2>')+4] + str(rotated_coors[i][2][1]) + object[object.find('</y2>') :]
                    object = object[ : object.find('<x3>')+4] + str(rotated_coors[i][3][0]) + object[object.find('</x3>') :]
                    object = object[ : object.find('<y3>')+4] + str(rotated_coors[i][3][1]) + object[object.find('</y3>') :]
                    _contents=_contents + '<object>' + object
                fw.write(_contents)  




# 计算旋转后的box角点坐标
def cal_rot_box(x,y,M):
    rot_x = x*M[0][0]+y*M[0][1]+M[0][2]
    rot_y = x*M[1][0]+y*M[1][1]+M[1][2]
    return rot_x,rot_y



# 功能：读取xml文件返回坐标和宽高信息（这里以四个点8坐标为例，可自定义修改）
# 传入：单个xml文件路径
# 返回：图像w,h，坐标信息(一维array，每个元素是包含八个坐标信息的list)------以后可根据自己需要自定义修改
# 注意：读取的是字符串形式，全部转化为int型返回，便于后续处理
def read_xml(xml_path):
    with open(xml_path,'r') as f:        
        contents=f.read()
        objects=contents.split('<object>')  # 注意分割之后'<object>'就没了，后面write要补上
        num=len(objects)-1  # 第一个是信息头
        assert num > 0, 'No object found in ' + xml_path 

        coor=[]        
        w = objects[0][objects[0].find('<width>')+7 : objects[0].find('</width>')]
        h = objects[0][objects[0].find('<height>')+8 : objects[0].find('</height>')]
        objects.pop(0)
        for object in objects:
            x0 = object[object.find('<x0>')+4 : object.find('</x0>')]
            y0 = object[object.find('<y0>')+4 : object.find('</y0>')]
            x1 = object[object.find('<x1>')+4 : object.find('</x1>')]
            y1 = object[object.find('<y1>')+4 : object.find('</y1>')]
            x2 = object[object.find('<x2>')+4 : object.find('</x2>')]
            y2 = object[object.find('<y2>')+4 : object.find('</y2>')]
            x3 = object[object.find('<x3>')+4 : object.find('</x3>')]
            y3 = object[object.find('<y3>')+4 : object.find('</y3>')]
            coor.append(np.array([int(x0),int(y0),int(x1),int(y1),int(x2),int(y2),int(x3),int(y3)]))        # 注意每个array元素是一组八个坐标

        return int(w),int(h),coor 
        




def transform():

    img_src = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/img_src'
    img_dst = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/img_dst'
    xml_src = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_src'
    xml_dst = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_dst'
    
    img_files= os.listdir(img_src) 
    for count,img_file in enumerate(img_files):
        img=cv2.imread(img_src+'/'+img_file)  # BGR
        assert img is not None, 'Failed to load this pic: ' + img_file

        # ------------图像增强变换部分----------------
        # Hsv(img,img_file,img_dst,xml_src,xml_dst)
        # Flip(img,img_file,img_dst,xml_src,xml_dst)
        # Noise(img,img_file,img_dst,xml_src,xml_dst,mode='Gausssian')
        # Blur(img,img_file,img_dst,xml_src,xml_dst)
        # Rot_Scale(img,img_file,img_dst,xml_src,xml_dst)
        Affine(img,img_file,img_dst,xml_src,xml_dst)

        # 打印进度条
        print('\rprogress bar :  {:.2f}%   '.format(count*100/len(img_files)),end='')








if __name__ == "__main__":
    transform()
