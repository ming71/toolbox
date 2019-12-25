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
    Reg = []
    Obj = []
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
            # yolov3的格式为：Epoch,gpu_mem,Reg,obj,cls,total,targets,img_size,P,R,mAP,F1,val Reg,val Objectness,val Classification
            epoch,_,obj,cls,reg,total,_,_,p,r,map,f1,*_ = line
            epoch = int(epoch.split('/')[0])    
            total = float(total)
            p   = float(p)
            r   = float(r)
            map = float(map)
            f1  = float(f1)
            reg = float(reg)
            obj = float(obj)
            Epoch.append(epoch)
            Loss.append(total)
            Reg.append(reg)
            Obj.append(obj)
            Precision.append(p)
            Recall.append(r)
            mAP.append(map)
            F1.append(f1)
    info = dict(epoch = Epoch, loss = Loss, reg = Reg,obj = Obj, precision = Precision, recall = Recall, mAP = mAP)
    return info
        


def display_settings(xlabel='',ylabel='',xlim=None,ylim=None,xticks=None,yticks=None,info=None,grid=True):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:    plt.xlim(xlim)  # 显示范围  eg. ylim=(0,1)
    if ylim is not None:    plt.ylim(ylim) 
    if xticks is not None:  plt.xticks(xticks)
    if yticks is not None:  plt.yticks(yticks)  # 刻度 eg. yticks = np.arange(0, 1, 0.1)
    if grid:                plt.grid(1)
    if len(info) > 1:       plt.legend()  




# 画图
def draw(infos, mode=['mAP','loss','precision','recall','P-R'], label=[]):
    # 显示多属性时手动排版
    if len(mode) < 4:
        row = 1
        col = len(mode)
    elif len(mode) == 4:
        row = 2
        col = 2 
    else:
        row = 2
        col = 3 
        print('大于4张图的，自己手动布局一下matplotlib的显示方式，设置row和cols即可')
    
    plt.figure(figsize=(12, 8),num='log') 

    # 调整图片生成的位置（距左上xy分别480 110）
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+400+110")

    color = ['red','C0','C1','C2','violet','brown','m','teal','blue','orange','cyan']
    for cnt,info in enumerate(infos):
        for i,type in enumerate(mode):
            if type == 'loss' :
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['loss'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='loss',ylim=(0,1),info=infos,grid=True)

            elif type == 'mAP':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['mAP'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='mAP',ylim=(0,1),yticks=np.arange(0, 1, 0.1),info=infos,grid=True)
                

            elif type =='precision':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['precision'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='precision',ylim=(0,1),yticks=np.arange(0, 1, 0.1),info=infos,grid=True)           

            elif type =='recall':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['recall'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='recall',ylim=(0,1),yticks=np.arange(0, 1, 0.1),info=infos,grid=True) 

            elif type =='reg':
                plt.subplot(row, col, i+1)
                plt.plot(info['epoch'], info['reg'],color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt])
                display_settings('epoch','reg',info=info,grid=True,ylim=(0,0.2))
            
            elif type =='obj':
                plt.subplot(row, col, i+1)
                plt.plot(info['epoch'], info['obj'],color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt])
                display_settings('epoch','obj',info=info,grid=True,ylim=(0,0.2))
            else:
                print('Unexpected type of mode!')
            

    plt.show()



if __name__ == "__main__":

    results = [ '/py/rotated-yolo/results.txt',
                 '/py/rotated-yolo/result/baseline.txt',
                 '/py/rotated-yolo/result/ga.txt',
                 '/py/rotated-yolo/result/dcn.txt',
                 '/py/rotated-yolo/result/orn8.txt',
                 '/py/rotated-yolo/result/se.txt',
            ]

    labels = [os.path.splitext(os.path.split(x)[1])[0] for x in results]
    infos = [get_info(res) for res in results]

    draw(infos, 
          mode=['mAP','precision','recall','loss','reg','obj'],  
          label=labels)

