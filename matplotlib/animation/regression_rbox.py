import sys
import math
import torch
import random
import numpy as np
import torch.nn.functional as F  
import matplotlib.pyplot as plt  
from riou_python import RIoU

def setup_seed(seed):
	torch.manual_seed(seed)					 # CPU
	torch.cuda.manual_seed_all(seed)	   # GPU	
	np.random.seed(seed)						 # numpy
	random.seed(seed)							  # random
	torch.backends.cudnn.deterministic = True	# cudnn

setup_seed(20)


def plot_polygon(ax, polygon, color='g'):
    rect = plt.Polygon(polygon, color = color, alpha = 0.5 )
    ax.add_patch(rect)

def rbox2poly(box, is_Radian=False):
    if isinstance(box, list):
        box = np.array(box).astype('float32')
    out_box = box.copy()
    if not is_Radian:
        out_box[4] = out_box[4] * 3.1415926 / 180.0
    cs = np.cos(out_box[4])
    ss = np.sin(out_box[4])
    w = out_box[2] - 1
    h = out_box[3] - 1
    ## change the order to be the initial definition
    x_ctr = out_box[0]
    y_ctr = out_box[1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)
    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)
    polys = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    return polys/1000

def box_encode(box):
    encode_box = box.clone()
    encode_box[:4] = torch.log(box[:4])
    encode_box[4] = torch.tan(box[4] / 180.0 * np.pi)
    return encode_box

def box_decode(box):
    decode_box = box.clone()
    decode_box[:4] = torch.exp(box[:4])
    decode_box[4] = torch.atan(box[4]) * 180.0 / np.pi
    return decode_box


class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.predict = torch.nn.Linear(n_hidden, n_output) 
 
    def forward(self, x):
        x = F.relu(self.hidden(x)) 
        x = self.predict(x)  
        return x
 
 
class Loss(object):
    def __init__(self, func='mse'):
        self.func = func

    def __call__(self, input, target):
        if self.func == 'mse':
            return self.mse_loss(input, target)
        if self.func == 'l1':
            return self.l1_loss(input, target)
        elif self.func == 'smoothl1':
            return self.smoothl1_loss(input, target)
        elif self.func == 'l2':
            return self.l2_loss(input, target)
        elif self.func == 'bi':
            return self.boundary_invariant_loss(input, target)
        else:
            raise NotImplementedError
    
    def mse_loss(self, input, target):
        mse = torch.nn.MSELoss() 
        return mse(input, target) 

    def l1_loss(self, input, target):
        return abs(input-target).mean()

    def smoothl1_loss(self, input, target):
        size_average = True
        beta=1. / 9
        diff = abs(input - target)
        loss = torch.where(
            diff < 0.5,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
        return loss.mean()

    def l2_loss(self, input, target):
        return (abs(input-target)**2).mean()

    def wh_iou(self, input, target):
        inter = torch.min(input[0] ,target[0]) * torch.min(input[1] ,target[1])
        union = input[0] * input[1]  + target[0] * target[1] - inter
        areac = torch.max(input[0] ,target[0]) * torch.max(input[1] ,target[1])
        hiou_loss = 1 - inter / (union + 1e-6) + (areac - union) / (areac + 1e-6)
        return hiou_loss



if __name__ == "__main__":
    # canvas: 1000x1000
    gt_box = [500, 500, 30, 800, 179]
    # gt_box = [500, 500, 800, 500, 60]
    init_box = torch.zeros(5)
    center_loss = []
    geometry_loss  = []
    total_loss = []
    
    total = 550
    model = Net(n_feature=5, n_hidden=10, n_output=5)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  
    # func = 'mse'
    # func = 'smoothl1'
    loss_func = Loss(func=func)  

    # plot regression process
    fig, ax = plt.subplots()
    plt.ion()
    for t in range(total): 
        print(t)
        prediction = model(init_box)  
        if func == 'bi':
            loss_items = loss_func(box_decode(prediction), torch.Tensor(gt_box)) 
            loss = loss_items.sum()
            center_loss.append(loss_items[0].data.item())
            geometry_loss.append(loss_items[1].data.item())
        else:
            loss = loss_func(prediction, box_encode(torch.Tensor(gt_box))) 
        total_loss.append(loss.data.item())
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        if t % 1 == 0:
            plt.cla()  
            plot_polygon(ax, rbox2poly(gt_box).reshape(4,2), 'r')
            pred = box_decode(prediction)
            plot_polygon(ax, rbox2poly(pred.data.numpy()).reshape(4,2))
            plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)
    plt.ioff()

    # plot loss curve
    figure = plt.figure(num=1, figsize=(10, 8),dpi=80) 
    if func == 'bi':
        plt.ylim((0, 1))
        plt.plot(range(total), center_loss, label = 'center_loss')
        plt.plot(range(total), geometry_loss, label = 'geometry_loss')
        plt.plot(range(total), total_loss, label = 'total_loss')
    else:
        plt.plot(range(total), total_loss, label = func)
    plt.legend(loc='best')
    plt.show()

