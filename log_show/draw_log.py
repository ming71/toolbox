#  ming71
#  2019.10.21

#  将打印的训练log文件读取数据并显示;支持多模型比较
#  自定义：info参数部分和信息提取部分，以及draw的额外参数

import os
import matplotlib.pyplot as plt
import numpy as np



# 通过此函数返回需要的信息，根据日志文件自定义
def get_info(log_path):
    Epoch = []
    Loss = []
    Precision = []
    Recall = []
    mAP = []
    F1 = []
    with open(log_path,'r') as f:
        contents = f.read()
        lines = contents.split('\n')[:-1]   # 分割的最后一个是\n，需要剔除（验证一下）
        for line in lines:
            line = [i for i in filter(None,line.split(" "))]    # 将行的空格去掉分割成一个个字符
            # 下面的信息提取自定义(下面以yolov3为例)
            # yolov3的格式为：Epoch,gpu_mem,GIoU,obj,cls,total,targets,img_size,P,R,mAP,F1,val GIoU,val Objectness,val Classification
            epoch,_,_,_,_,total,_,_,p,r,map,f1,*_ = line
            epoch = int(epoch.split('/')[0])    
            total = float(total)
            p = float(p)
            r = float(r)
            map = float(map)
            f1 = float(f1)
            Epoch.append(epoch)
            Loss.append(total)
            Precision.append(p)
            Recall.append(r)
            mAP.append(map)
            F1.append(f1)
    info = dict(epoch = Epoch, loss = Loss, precision = Precision, recall = Recall, mAP = mAP)
    return info
        


def display_settings(xlabel='',ylabel='',xlim=None,ylim=None,xticks=None,yticks=None,info=None,grid=True):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:    plt.xlim(xlim)
    if ylim is not None:    plt.ylim(ylim) 
    if xticks is not None:  plt.xticks(xticks)
    if yticks is not None:  plt.yticks(yticks)  # eg. yticks = np.arange(0, 1, 0.1)
    if grid:                plt.grid()
    if len(info) > 1:       plt.legend()   



# 画图
def draw(infos, mode=['mAP','loss','Precision','Recall','P-R'], label=[]):
    # 显示多属性时手动排版
    if len(mode) < 4:
        row = 1
        col = len(mode)
    elif len(mode) == 4:
        row = 2
        col = 2
    else:
        print('大于4张图的，自己手动布局一下matplotlib的显示方式，设置row和cols即可')
    
    plt.figure(figsize=(10, 8),num='log') 

    # 调整图片生成的位置（距左上xy分别480 110）
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+480+110")

    color = ['C0','C1','C2','teal','red','violet']
    for cnt,info in enumerate(infos):
        for i,type in enumerate(mode):
            if type == 'loss' :
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['loss'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='loss',info=infos,grid=True)

            elif type == 'mAP':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['mAP'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='mAP',ylim=(0,1),yticks=np.arange(0, 1, 0.1),info=infos,grid=True)

            elif type =='Precision':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['precision'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='precision',ylim=(0,1),info=infos,grid=True)           

            elif type =='Recall':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['recall'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='recall',ylim=(0,1),info=infos,grid=True) 

            elif type =='P-R':
                assert 'Recall' in mode and 'Precision' in mode ,'Recall or Precision is lost,can not draw P-R cruve!!'
                plt.subplot(row, col, i+1)
                plt.plot(info['recall'], info['precision'],color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt])
                display_settings('recall','precision',info=info,grid=True)
            else:
                print('Unexpected type of mode!')

    plt.show()



if __name__ == "__main__":


    # 画一条曲线
    # log_path = r'/py/yolov3/results.txt'
    # info = get_info(log_path)
    # draw([info],mode=['loss','mAP'],label=['yolo'])

    # 画多条曲线对比
    log_path1 = r'/py/yolov3/weights/single-ship/results.txt'
    log_path2 = r'/py/matrix-yolo/weights/single-ship/v1/results.txt'
    log_path3 = r'/py/matrix-yolo/weights/single-ship/v1-unshare/results.txt'
    log_path4 = r'/py/matrix-yolo/weights/single-ship/v2/results.txt'
    log_path5 = r'/py/matrix-yolo/results.txt'


    info1 = get_info(log_path1)
    info2 = get_info(log_path2)
    info3 = get_info(log_path3)
    info4 = get_info(log_path4)
    info5 = get_info(log_path5)

    # draw([info1,info2],mode=['mAP'],label=('yolo','matrix-yolo-v1','matrix-yolo-v2'))
    draw([info1,info4,info5], mode=['mAP','Precision','Recall','loss'],    \
                              label=('yolo','matrix-yolo-v1','matrix-yolo-v1-unshare','matrix-yolo-v2')  )