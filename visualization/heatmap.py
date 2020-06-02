'''
### 可视化attention map，但是失败了，特征很随机，可能SA的空域对齐本身就是个伪命题？
import os
import cv2
import torch
import shutil
import argparse
import numpy as np
import torch.nn.functional as F
from datasets import *
from models.stela import STELA
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption, hyp_parse


cls_attens = []
reg_attens = []
def get_cls_attention(module, fea_in, fea_out):
    cls_attens.append(fea_out[1])
    return None

def get_reg_attention(module, fea_in, fea_out):
    reg_attens.append(fea_out[1])
    return None


DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'HRSC2016': HRSCDataset,
            }

def heatmap(args):
    hyps = hyp_parse(args.hyp)
    ds = DATASETS[args.dataset](level = 1)
    model = STELA(backbone=args.backbone, hyps=hyps)
    root_dir = 'outputs/heatmaps'
    # hook
    model.cls_attention.register_forward_hook(get_cls_attention)
    model.reg_attention.register_forward_hook(get_reg_attention)

    if args.weight.endswith('.pth'):
        chkpt = torch.load(args.weight)
        # load model
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        print('load weight from: {}'.format(args.weight))
    model.eval()

    if  os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.mkdir(root_dir)
    
    ims_list = [x for x in os.listdir(args.ims_dir) if is_image(x)]
    for idx, im_name in enumerate(ims_list):
        # 制定hook提取attention
        cams = []
        rams = []
        im_path = os.path.join(args.ims_dir, im_name)   # 单张照片绝对路径
        s=''
        s += 'image %g/%g %s: ' % (idx, len(ims_list), im_path)
        src = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        cls_dets = im_detect(model, im, target_sizes=args.target_size)
        for cls_atten in cls_attens[-1]:
            _cam = cv2.resize(cls_atten.squeeze(0).cpu().numpy().transpose(2,1,0), src.transpose(1,0,2).shape[:-1])
            cams.append(_cam*255)
        for reg_atten in reg_attens[-1]:
            _ram = cv2.resize(reg_atten.squeeze(0).cpu().numpy().transpose(2,1,0), src.transpose(1,0,2).shape[:-1])
            rams.append(_ram*255)          
        resized_cams = []
        resized_rams = []
        ### 多fp求取平均  
        # cam = np.array(np.stack(cams,0).mean(0),dtype='uint8') 
        # ram = np.array(np.stack(rams,0).mean(0),dtype='uint8') 
        # heatmap_cls = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # heatmap_reg = cv2.applyColorMap(ram, cv2.COLORMAP_JET)
        # cls_res = cv2.addWeighted(heatmap_cls, 0.5, src, 0.5, 0)
        # reg_res = cv2.addWeighted(heatmap_reg, 0.5, src, 0.5, 0)
        # filename = os.path.splitext(os.path.split(im_path)[1])[0]
        # cv2.imwrite(os.path.join(root_dir,filename+'_cls.jpg'), cls_res)
        # cv2.imwrite(os.path.join(root_dir,filename+'_reg.jpg'), reg_res)
        ### 每张特征图单独可视化
        for layer_id, (cam,ram) in enumerate(zip(cams,rams)):
            cam = np.array(cam,dtype='uint8') 
            ram = np.array(ram,dtype='uint8') 
            heatmap_cls = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            heatmap_reg = cv2.applyColorMap(ram, cv2.COLORMAP_JET)
            cls_res = cv2.addWeighted(heatmap_cls, 0.5, src, 0.5, 0)
            reg_res = cv2.addWeighted(heatmap_reg, 0.5, src, 0.5, 0)
            filename = os.path.splitext(os.path.split(im_path)[1])[0]
            cv2.imwrite(os.path.join(root_dir,filename+'_cls_{}.jpg'.format(layer_id)), cls_res)
            cv2.imwrite(os.path.join(root_dir,filename+'_reg_{}.jpg'.format(layer_id)), reg_res)


# 注：由于是hook提取的attens，建议sample少放点图！
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='weights/last.pth')
    parser.add_argument('--dataset', type=str, default='HRSC2016')
    
    parser.add_argument('--ims_dir', type=str, default='HRSC2016/Minitest/AllImages')  
#     parser.add_argument('--ims_dir', type=str, default='samples')
    parser.add_argument('--target_size', type=int, default=416)
    heatmap(parser.parse_args())

'''

### 一个简单的小例子
# 下面的weight图是原图的直接灰度化加权，实际用的时候可以把fp，attention mask等作为weight加权
# 实际使用还涉及数据格式的转换等问题，参考上面的程序
import cv2

img = cv2.imread('woof_meow.jpg', 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
res = cv2.addWeighted(heatmap_img, 0.7, img, 0.3, 0)
cv2.imshow('heatmap', res)
cv2.waitKey(0)