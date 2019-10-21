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
    path = r'/py/matrix-yolo/cam'
    figs = (5,3)    # 设置行列数
    row_name = 'fp'   # 设置行列图像编号名(如fp,anchor)
    col_name = 'anchor'
    
    
    plt.figure(figsize=(100, 100))
    # 行列设置
    fig1_rows_1 = ['10.jpg','11.jpg','12.jpg']
    fig1_rows_2 = ['70.jpg','71.jpg','72.jpg']
    fig1_rows_3 = ['80.jpg','81.jpg','82.jpg']
    fig1_rows_4 = ['90.jpg','91.jpg','92.jpg']
    fig1_rows_5 = ['100.jpg','101.jpg','102.jpg']


    subpath = [fig1_rows_1, fig1_rows_2, fig1_rows_3,fig1_rows_4,fig1_rows_5]
    outpath = abs_path(path,subpath)

    cnt = 0
    for r,row in enumerate(outpath):
        for c,img_path in enumerate(row):
            print(img_path)
            cnt += 1
            img = cv2.imread(img_path,1)
            img = img[...,::-1]
            ax = plt.subplot(figs[0], figs[1], cnt)
            ax.set_xlabel(img_path)
            plt.imshow(img)

    plt.show()
