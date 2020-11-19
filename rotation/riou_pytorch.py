import torch
import math
import numpy as np
import matplotlib.pyplot as plt

###### shapely skewiou for check ######
import cv2
import shapely
import numpy as np 
from shapely.geometry import Polygon,MultiPoint
def shapely_iou(box1, box2):
    rbox1 = rbox2points(box1).reshape(4, 2)
    rbox2 = rbox2points(box2).reshape(4, 2)
    poly1 = Polygon(rbox1).convex_hull
    poly2 = Polygon(rbox2).convex_hull
    
    union_poly = np.concatenate((rbox1,rbox2))  
    if not poly1.intersects(poly2): 
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area 
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou= 0
            iou=float(inter_area) / union_area
            
        except shapely.geos.TopologicalError:
            print("shapely.geos.TopologicalError occured, iou set to 0")
            iou = 0
    return iou

def rbox2points(box):
    cx,cy,w,h,a = box
    points = cv2.boxPoints(((cx,cy),(w,h),a))
    return points
#######################################


class Rec():
    def __init__(self,center,l,w,theta):
        # 角度为长l与水平边的夹角，逆时针方向，弧度角
        self.center = center
        self.l = l
        self.w = w
        self.theta = theta
        # 计算出四个顶点的坐标
        l_vec = torch.stack([l/2*torch.cos(theta),l/2*torch.sin(theta)])
        w_vec = torch.stack([w/2*torch.cos(3/2*math.pi+theta),w/2*torch.sin(3/2*math.pi+theta)])
        # 将四个顶点存起来
        self.points = torch.stack([
             l_vec + w_vec + self.center,
             l_vec - w_vec + self.center,
            -l_vec - w_vec + self.center,
            -l_vec + w_vec + self.center
        ])
    
    def area(self):
        area = self.l * self.w
        return area

    def draw(self):
        rec_points = torch.cat((self.points, self.points[0].unsqueeze(0)),0)   
        plt.plot(rec_points[:,0],rec_points[:,1],'-')

def point2rec(point,rec):
    # 判断一个点是否在一个矩形中
    rec_points = torch.cat((rec.points, rec.points[0].unsqueeze(0)),0)

    flag = None
    for i in range(4):
        # 利用叉乘判断一个点是否在一个矩形内
        k1 = rec_points[i+1,:]-rec_points[i,:]
        k2 = point-rec_points[i+1,:]
        multip = k1[0] * k2[1] - k1[1] * k2[0]
        if flag is None:
            flag = multip
        else:
            if flag * multip < -1e-6:
                return False
    return True

def line2line(line1,line2):
    xa,ya = line1[0,0],line1[0,1]
    xb,yb = line1[1,0],line1[1,1]
    xc,yc = line2[0,0],line2[0,1]
    xd,yd = line2[1,0],line2[1,1]
    # 判断两条直线是否相交，矩阵行列式计算
    a = torch.stack([
        torch.stack([xb-xa,-(xd-xc)]),
        torch.stack([yb-ya,-(yd-yc)])
    ])
    delta = torch.det(a)
    # 不相交,返回两线段
    if abs(delta) < 1e-6:
        return []
    # 求两个参数lambda和miu
    c = torch.stack([
        torch.stack([xc-xa,-(xd-xc)]),
        torch.stack([yc-ya,-(yd-yc)])
    ])
    d = torch.stack([
        torch.stack([xb-xa,xc-xa]),
        torch.stack([yb-ya,yc-ya])
    ])
    lamb = torch.det(c)/delta
    miu = torch.det(d)/delta
    # 相交
    if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
        x = xc + miu*(xd-xc)
        y = yc + miu*(yd-yc)
        return torch.stack([x, y])
    # 相交在延长线上
    else:
        return []

def line2rec(line,rec):
    # 线段与矩形的交点
    rec_points = torch.cat((rec.points, rec.points[0].unsqueeze(0)),0)   # 形成闭合的点列 
    cross_points=[]
    for i in range(4):
        cross_point=line2line(line,rec_points[i:i+2,:])
        if len(cross_point) != 0:
            if len(cross_points) == 0:
                cross_points = cross_point.unsqueeze(0)
            else:
                cross_points = torch.cat([cross_points, cross_point.unsqueeze(0)], 0)

    # 把在矩形内的端点也作为交点
    if len(cross_points) == 1:
        if point2rec(line[0,:],rec):
            cross_points = torch.cat((cross_points, line[0,:].unsqueeze(0)),0)
        else:
            cross_points = torch.cat((cross_points, line[1,:].unsqueeze(0)),0)

    # 如果一条线与矩形没有交点，需要考虑点在矩形内部
    if len(cross_points) == 0 and point2rec(line[0,:],rec) and point2rec(line[1,:],rec):
        cross_points = torch.cat((cross_points, line[0,:].unsqueeze(0)),0)
        cross_points = torch.cat((cross_points, line[1,:].unsqueeze(0)),0)
    return cross_points

def rec2rec(rec1,rec2):
    # 矩形与矩形的交点，形成多边形
    rec1_points = torch.cat((rec1.points, rec1.points[0].unsqueeze(0)),0)   # 形成闭合的点列 
    rec2_points = torch.cat((rec2.points, rec2.points[0].unsqueeze(0)),0)   # 形成闭合的点列 

    cross_points = []
    for i in range(4):
        # 求出矩形1每条边与矩形2的交点
        cross_point = line2rec(rec1_points[i:i+2,:], rec2)
        if len(cross_point) != 0:
            if len(cross_points) == 0 :
                cross_points = cross_point
            else:
                cross_points = torch.cat([cross_points, cross_point], 0)

    # 要求两次,因为关注的不只是相交,还有顶点落在矩形内的情况
    for i in range(4):
        # 求出矩形2每条边与矩形1的交点
        cross_point = line2rec(rec2_points[i:i+2,:], rec1)
        if len(cross_point) != 0:
            if len(cross_points) == 0 :
                cross_points = cross_point
            else:
                cross_points = torch.cat([cross_points, cross_point], 0)
    return cross_points



def refine_points(points):
    if len(points) == 0:
        return [], 0
    new_points = points[:2]
    rest_points = points[2:]
    # 将重复的点删除，按照顺序形成多边形
    # 跳出循环的条件是收尾点重合构成封闭多边形(每个交点和矩形内的点必然出现两次)
    while ((new_points[-1, :]-new_points[0, :])**2).sum() > 1e-6:   
        tmp_point = new_points[-1,:]
        ind = (((rest_points - tmp_point)**2).sum(1) < 1e-6).nonzero().squeeze()
        rest_points = rest_points[torch.arange(rest_points.size(0))!= ind]   
        if ind % 2 == 0:   
            new_points = torch.cat((new_points,rest_points[ind].unsqueeze(0)),0)
            rest_points = rest_points[torch.arange(rest_points.size(0))!= ind] 
        else:
            new_points = torch.cat((new_points,rest_points[ind-1].unsqueeze(0)),0)
            rest_points = rest_points[torch.arange(rest_points.size(0))!= (ind-1)] 
    
    # 求面积
    S = 0 
    for i in range(1,new_points.shape[0]-2):
        a = torch.norm(new_points[0,:]-new_points[i,:])
        b = torch.norm(new_points[i,:]-new_points[i+1,:])
        c = torch.norm(new_points[0,:]-new_points[i+1,:])
        S += tri_area(a,b,c)

    return new_points,S

def tri_area(a,b,c):
    # 三角形面积
    # 计算半周长
    s = (a + b + c) / 2
    # 计算面积
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area

def RIoU(box1, box2, plot=False):
    # 定义两个矩形
    Rec1 = Rec(center=torch.stack([box1[0], box1[1]]),l=box1[2],w=box1[3],theta=box1[4]/180*3.14159)
    Rec2 = Rec(center=torch.stack([box2[0], box2[1]]),l=box2[2],w=box2[3],theta=box2[4]/180*3.14159)
    if plot:
        Rec1.draw()
        Rec2.draw()

    # 求出矩形1与矩形2的交点
    cross_points = rec2rec(Rec1,Rec2)
    
    # 将重复的交点去掉，并顺序表示
    points, inter = refine_points(cross_points)
    union = Rec1.area() + Rec2.area() - inter
    iou = inter / union 


    if len(points) != 0 and plot:
        # 只有当有交点时才画出相交面积
        plt.plot(points[:,0],points[:,1],'ro-')
        plt.fill(points[:,0],points[:,1], facecolor='r',alpha=0.5)

        plt.title('IoU={:.4f}'.format(iou))
        plt.axis('equal')
        plt.show()

    return iou
                                
if __name__ == "__main__":
    box1 = torch.Tensor([30,100,200,120,0])
    box2 = torch.Tensor([200,150,200,120,45])
    # checkout result
    skewiou = shapely_iou(box1, box2)
    print(skewiou)
    riou = RIoU(box1, box2, True)
    print(riou)
