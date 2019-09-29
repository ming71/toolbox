#  将打印的训练log文件读取数据并显示
#  自定义：info参数部分和信息提取部分，以及draw的额外参数

import os
import matplotlib.pyplot as plt

# 需要的信息
def infos_init():
    global Epoch 
    global Loss
    global Precision 
    global Recall 
    global mAP 
    global F1 
    Epoch = []
    Loss = []
    Precision = []
    Recall = []
    mAP = []
    F1 = []



# 通过此函数返回需要的信息，根据日志文件自定义
def get_info(log_path):
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
        
# 画图
def draw(info,mode=['loss','mAP']):
    if len(mode) < 4:
        row = 1
        col = len(mode)
    elif len(mode) == 4:
        row = 2
        col = 2
    else:
        print('自己手动布局一下matplotlib的显示方式，设置row和cols即可')

    for i,type in enumerate(mode):
        if type == 'loss' :
            plt.figure(1) 
            plt.subplot(row, col, i+1) 
            plt.plot(info['epoch'], info['loss'], color="C0", linestyle="-",  linewidth=1) 
            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel("loss")
        elif type == 'mAP':
            plt.figure(1) 
            plt.subplot(row, col, i+1) 
            plt.plot(info['epoch'], info['mAP'], color="C2", linestyle="-",  linewidth=1) 
            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel("mAP")
        else:
            print('Unexpected type of mode!')
    
    plt.show()



if __name__ == "__main__":
    log_path = r'/py/yolov3/results.txt'

    infos_init()
    info = get_info(log_path)
    draw(info,mode=['loss','mAP'])
