import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob


def abs_path(parentpath,subpath):
    out = []
    if isinstance(subpath,list):
        for _subpath in subpath:
            out.append([os.path.join(parentpath,p) for p in _subpath])
    else :
        print('worng type of subpath')
    return out

if __name__ == "__main__":
    path = r'/py/yolov3/cam/rlt'
    figs = (3,3)    # 设置行列
    
    plt.figure(figsize=(10, 10))
    # 行列设置
    fig1_rows_1 = ['cls_0_0_(1).jpg','cls_0_1_(1).jpg','cls_0_2_(1).jpg']
    fig1_rows_2 = ['cls_1_0_(1).jpg','cls_1_1_(1).jpg','cls_1_2_(1).jpg']
    fig1_rows_3 = ['cls_2_0_(1).jpg','cls_2_1_(1).jpg','cls_2_2_(1).jpg']
    # fig1_rows_1 = ['bam_0_0_cls2.jpg','bam_0_1_cls2.jpg','bam_0_2_cls2.jpg']
    # fig1_rows_2 = ['bam_1_0_cls2.jpg','bam_1_1_cls2.jpg','bam_1_2_cls2.jpg']
    # fig1_rows_3 = ['bam_2_0_cls2.jpg','bam_2_1_cls2.jpg','bam_2_2_cls2.jpg']

    subpath = [fig1_rows_1, fig1_rows_2, fig1_rows_3]
    outpath = abs_path(path,subpath)

    cnt = 0
    for r,row in enumerate(outpath):
        for c,img_path in enumerate(row):
            cnt += 1
            img = cv2.imread(img_path,1)
            img = img[...,::-1]
            ax = plt.subplot(figs[0], figs[1], cnt)
            ax.set_xlabel('fp:{}  anchor:{} '.format(r,c))
            plt.imshow(img)

    plt.show()
