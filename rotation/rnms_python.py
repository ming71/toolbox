import torch
import cv2
import numpy as np
import matplotlib.pyplot  as plt

def cross(a,b):
    '''平面向量的叉乘'''
    x1,y1 = a
    x2,y2 = b
    return x1 * y2 - x2 * y1
def line_cross(line1,line2):
    '''判断两条线段是否相交,并求交点'''
    a,b = line1
    c,d = line2
    if a.device != c.device:
        return False
    
    # 两个三角形的面积同号或者其中一个为0（其中一条线段端点落在另一条线段上） ---> 不相交
    if cross(c - a,b - a) * cross(d - a,b - a) >= 0:
        return False
    if cross(b - c,d - c) * cross(a - c,d - c) >= 0:
        return False
    x1,y1 = a
    x2,y2 = b
    x3,y3 = c
    x4,y4 = d
    
    k = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) 
    if  k != 0:
        xp = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / k
        yp = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / k
    else:
        # 共线
        return False
    return xp,yp

def compare(a,b,center):
    '''
    对比a-center线段是在b-center线段的顺时针方向（True）还是逆时针方向（False）
    1. 通过叉乘积判断，积为负则a-center在b-center的逆时针方向，否则a-center在b-center的顺时针方向；
    2. 如果a,b,center三点共线，则按距离排列，距离center较远的作为顺时针位。

    原理：
    det = a x b = a * b * sin(<a,b>)
    其中<a,b>为a和b之间的夹角，意义为a逆时针旋转到b的位置所需转过的角度
    所以如果det为正，说明a可以逆时针转到b的位置，说明a在b的顺时针方向
    如果det为负，说明a可以顺时针转到b的位置，说明a在b的逆时针方向

    '''
    det = cross(a - center, b - center)
    if det > 0:
        return True
    elif det < 0:
        return False
    else:
        d_a = torch.sum((a - center) ** 2)
        d_b = torch.sum((b - center) ** 2)
        if d_a > d_b:
            return True
        else:
            return False

def quick_sort(box,left,right,center = None):
    '''快速排序'''
    if center is None:
        center = torch.mean(box,dim = 0)
    if left < right:
        q = partition(box,left,right,center)
        quick_sort(box,left,q - 1,center)
        quick_sort(box,q + 1,right,center)

def partition(box,left,right,center = None):
    '''辅助快排，使用最后一个元素将'''
    x = box[right]
    i = left - 1
    for j in range(left,right):
        if compare(x,box[j],center):
            i += 1
            temp = box[i].clone()
            box[i] = box[j]
            box[j] = temp
            # torch.Tensor不能使用下面的方式进行元素交换
            # box[i],box[j] = box[j],box[i]
    temp = box[i + 1].clone()
    box[i + 1] = box[right]
    box[right] = temp
    return i + 1

def inside(point,polygon):
    '''
    判断点是否在多边形内部
    原理：
    射线法
    从point作一条水平线，如果与polygon的焦点数量为奇数，则在polygon内，否则在polygon外

    为了排除特殊情况
    只有在线段的一个端点在射线下方，另一个端点在射线上方或者射线上的时候，才认为线段与射线相交
    '''
    x0,y0  = point
    # 做一条从point到多边形最左端位置的水平(y保持不变)射线
    left_line = torch.Tensor([[x0,y0],[torch.min(polygon,dim = 0)[0][0].item() - 1,y0]])
    lines = [[polygon[i],polygon[i+1]] for i in range(len(polygon) - 1)] + [[polygon[-1],polygon[0]]]
    ins = False
    for line in lines:
        (x1,y1),(x2,y2) = line
        if min(y1,y2) < y0 and max(y1,y2) >= y0:
            c = line_cross(left_line,line)
            if c and c[0] <= x0:
                ins = not ins
    return ins

def intersection(box1,box2):
    '''
    判断两个框是否相交，如果相交，返回重叠区域的顶点
    1. 求box1在box2内部的点；
    2. 求box2在box1内部的点；
    3. 求box1和box2的交点；
    4. 所有点构成重叠区域的多边形点集；
    5. 顺时针排序
    '''
    quick_sort(box1,0,len(box1) - 1)
    quick_sort(box2,0,len(box2) - 1)
    # 求重叠区域
    # 整理成线段
    lines1 = [[box1[i],box1[i + 1]] for i in range(len(box1) - 1)] + [[box1[-1],box1[0]]]
    lines2 = [[box2[i],box2[i + 1]] for i in range(len(box2) - 1)] + [[box2[-1],box2[0]]]
    cross_points = []
    # 交点
    for l1 in lines1:
        for l2 in lines2:
            c = line_cross(l1,l2)
            if c:
                cross_points.append(torch.Tensor(c).view(1,-1))
    # 求box1在box2内部的点
    for b in box1:
        if inside(b,box2):
            cross_points.append(b.view(1,-1))
    for b in box2:
        if inside(b,box1):
            cross_points.append(b.view(1,-1))
    if len(cross_points) > 0:
        cross_points = torch.cat(cross_points,dim = 0)
        quick_sort(cross_points,0,len(cross_points) - 1)
        return cross_points
    else:
        return None

def polygon_area(polygon):
    '''
    求多边形面积
    https://blog.csdn.net/m0_37914500/article/details/78615284 使用向量叉乘计算多边形面积，前提是多边形所有点按顺序排列
    '''
    lines = [[polygon[i],polygon[i+1]] for i in range(len(polygon) - 1)] + [[polygon[-1],polygon[0]]]
    s_polygon = 0.0
    for line in lines:
        a,b = line
        s_tri = cross(a,b)
        s_polygon += s_tri
    return s_polygon / 2

def intersection_of_union(box1,box2):
    '''
    iou = intersection(s_1,s_2) / (s_1 ＋ s_2 - intersection(s_1,s_2))
    '''
    quick_sort(box1,0,len(box1) - 1)
    quick_sort(box2,0,len(box2) - 1)
    s_box1 = torch.abs(polygon_area(box1))
    s_box2 = torch.abs(polygon_area(box2))
    cross_points = intersection(box1,box2)
    if cross_points is not None:
        # cv2.polylines(empty, [cross_points.data.numpy().astype(np.int32)], True, (0, 0, 255), 4)
        s_cross = torch.abs(polygon_area(cross_points))
    else:
        s_cross = torch.Tensor([[0]])
    iou = s_cross / (s_box1 + s_box2 - s_cross)
    return iou


def rnms(boxes,scores,score_thresh = 0,nms_thresh = 0.1):
    indices = torch.where(scores > score_thresh)[0]
    if len(indices) <= 1:
        return boxes[indices]
    boxes = boxes[indices]
    scores = scores[indices]
    keep_indices = []
    # 从大到小
    order = torch.argsort(scores).flip(dims = [0])
    while order.shape[0] > 0:
        i = order[0]
        keep_indices.append(i)
        not_overlaps = []
        for j in range(len(order)):
            if order[j] != i:
                iou = intersection_of_union(boxes[i],boxes[order[j]])
                if iou < nms_thresh:
                    not_overlaps.append(j)
        order = order[not_overlaps]
    keep_boxes = boxes[[i.item() for i in keep_indices]]
    return keep_boxes

