#  ming71

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
    Var = []
    Ref = []
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
            ## 根据log文件的条目设置好各个参数的位置，append到对应的list中
            # yolov3的格式为：Epoch, gpu_mem, Reg,obj, cls, total, targets, img_size, P, R, mAP, F1
            if len(line) == 12:
                epoch,_,obj,cls,reg,total,_,_,p,r,map,f1,*_ = line
                obj = float(obj)
                Obj.append(obj)

            # elif len(line) == 13:
            #     epoch,_,ref,cls,reg,var,total,_,_,p,r,map,f1,*_ = line
            #     var = float(var)
            #     ref = float(ref)
            #     Var.append(var)
            #     Ref.append(ref)
            total = float(total)
            F1.append(float(f1))
            Epoch.append(int(epoch.split('/')[0]))
            Loss.append(total)
            Reg.append(reg)
            Precision.append(float(p))
            Recall.append(float(r))
            mAP.append(float(map))
    info = dict(epoch = Epoch, loss = Loss, reg = Reg,obj = Obj,
                precision = Precision, recall = Recall, mAP = mAP, f1 = F1)
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
    
    # plt.figure(figsize=(12, 8),num='log') 
    plt.figure(num='log') 

    # 调整图片生成的位置（距左上xy分别480 110）
    mngr = plt.get_current_fig_manager()
    # mngr.window.wm_geometry("+400+110")

    color = ['red','C0','C1','C2','violet','brown','m','teal','blue','orange','cyan']
    for cnt,info in enumerate(infos):
        for i,type in enumerate(mode):
            if type == 'loss' :
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['loss'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='loss',ylim=(0,5),info=infos,grid=True)

            elif type == 'mAP':
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['mAP'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='mAP',ylim=(0,1),yticks=np.arange(0, 1, 0.1),info=infos,grid=True)

            elif type == 'f1':
                import ipdb; ipdb.set_trace()
                plt.subplot(row, col, i+1) 
                plt.plot(info['epoch'], info['f1'], color=color[cnt], linestyle="-",  linewidth=1, label=label[cnt]) 
                display_settings(xlabel='epoch',ylabel='F1',ylim=(0,1),yticks=np.arange(0, 1, 0.1),info=infos,grid=True)
                 

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

    root_dir = r'D:\研究生\工作\AnchorMatching\experiments\mAP'
    results = [ 
                's0_416_0.3',
                's0_416_0.5',
                's0_416_noassigner_0.3',
                's1_416_0.3-0.5'
            ]
    suffix = '.txt'
    filename = [os.path.join(root_dir, x+suffix) for x in results]

    labels = [os.path.splitext(os.path.split(x)[1])[0] for x in filename]
    infos = [get_info(res) for res in filename]

    draw(infos, 
          mode=['mAP','loss'],  
          label=labels)

