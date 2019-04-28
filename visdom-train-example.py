'''
note:
visdom三部曲：导入-创建env-画图
    import visdom
    vis = visdom.Visdom(env='MNIST')
    vis.line(X=np.array([i]),Y=np.array([sum_loss]),win='loss',update='append',name='lr-0.001',opts=dict(linecolor=np.array([[218,165,32]]),showlegend=True))
常用的是line工具，需要注意的地方：
1.颜色设置：array格式，shape[0]为1（为了兼容多种颜色输入），如果只是单线颜色设置可以为：np.array([[218,165,32]]
2.name设置了就不用在legend在命名了，不然不会更新
3.如果想要多次运行在一张图上更新线，需要更改的opt：颜色，name，直接运行即可如果是开多个类型的曲线，可以更改win多开窗口

'''

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import visdom
import numpy as np

from utils import LoadImagesAndLabels
#from utils import LoadImages
from model import *
from inference import test

import ipdb


def train(epoch,train_path,img_size,save_path,augmentation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataloader生成迭代器，tuple存储data和label    
    train_dataloader = LoadImagesAndLabels(train_path,img_size, augment=augmentation)
    
#    net = LeNet().to(device)
    net = Baseline_Net().to(device)
#    net = FC_Fused_Net().to(device)
#    net = ReLU_Aug_Net().to(device)
#    net = Inception_like_Net().to(device)
#    print(net)
    
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)
    for epoch in range(epoch):
        sum_loss = 0.0
#        ipdb.set_trace()
        for i, data in enumerate(train_dataloader):
            # data中img为tensor ，label tensor   
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
           
#            if i <15:
#                for param_group in optimizer.param_groups:
#                    param_group['lr'] = 0.08
#            elif i <46:
#                for param_group in optimizer.param_groups:
#                    param_group['lr'] = 0.01
#            elif i <300:
#                for param_group in optimizer.param_groups:
#                    param_group['lr'] = 0.005  
#            else:
#                for param_group in optimizer.param_groups:
#                    param_group['lr'] = 0.001          

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            print('[epoch:%d,batch: %d] loss: %.07f'
                      % (epoch + 1, i + 1, sum_loss ))
            vis.line(X=np.array([i]),Y=np.array([sum_loss]),win='loss',update='append',name='lr-0.001',opts=dict(linecolor=np.array([[218,165,32]]),showlegend=True))

            sum_loss = 0.0
            

            if i %1==0:
                torch.save(net, '%s/lr-adaptivenet_%03d.pth' % (save_path, i + 1))
                ac=test('/py/mnist/data/test_list.txt',28,'%s/lr-adaptivenet_%03d.pth' % (save_path, i + 1))
                vis.line(X=np.array([i]),Y=np.array([ac]),win='accurancy',update='append',name='lr-0.001',opts=dict(linecolor=np.array([[218,165,32]]),showlegend=True))
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=1, help='epoches') 
    parser.add_argument('--img_size', default=28, help='img size ')
    parser.add_argument('--train_path', default='/py/mnist/data/traintrain_list.txt', help="train_list_txt_path")  
    parser.add_argument('--save_path', default='/py/mnist/weights', help='path to save model')
    parser.add_argument('--augmentation', default=False, help='aug ')
    opt = parser.parse_args()
    print(opt)
    vis = visdom.Visdom(env='MNIST')
    train(opt.epoch,opt.train_path,opt.img_size,opt.save_path,opt.augmentation)


