import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形
from shapely.geometry import Polygon
import random
import math
import matplotlib.pyplot as plt
import cv2
import re

# 传入的坐标是[x1,y1,x2,y2,x3,y3,x4,y4]一维list
# 不用关注点的顺序，会自动编排，但是建议最好不要交叉，按照顺/逆时针输入

def skewiou(box1, box2,mode='iou',return_coor = False):
    box1=np.asarray(box1)
    box2=np.asarray(box2)
    a=np.array(box1).reshape(4, 2)   
    b=np.array(box2).reshape(4, 2)
    # 所有点的最小凸的表示形式，四边形对象，会自动计算四个点，最后顺序为：左上 左下  右下 右上 左上
    poly1 = Polygon(a).convex_hull  
    poly2 = Polygon(b).convex_hull

    # 异常情况排除
    if not poly1.is_valid or not poly2.is_valid :
        print('formatting errors for boxes!!!! ')
        return 0
    if  poly1.area == 0 or  poly2.area  == 0 :
        return 0

    inter = Polygon(poly1).intersection(Polygon(poly2)).area
    if   mode == 'iou':
        union = poly1.area + poly2.area - inter
    elif mode =='tiou':
        union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
        union = MultiPoint(union_poly).convex_hull.area
        coors = MultiPoint(union_poly).convex_hull.wkt
    elif mode == 'giou':
        union_poly = np.concatenate((a,b))   
        union = MultiPoint(union_poly).envelope.area
        coors = MultiPoint(union_poly).envelope.wkt
    elif mode== 'r_giou':
        union_poly = np.concatenate((a,b))   
        union = MultiPoint(union_poly).minimum_rotated_rectangle.area
        coors = MultiPoint(union_poly).minimum_rotated_rectangle.wkt
    else:
        print('incorrect mode!')

    if union == 0:
        return 0
    else:
        if return_coor:
            return inter/union,coors
        else:
            return inter/union



# 输入的box是xywha
def get_rotated_coors(box):
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 

    r_box=[x0,y0,x1,y1,x2,y2,x3,y3]
    return r_box




def vis_unionboxes(box,img_path):
    degree = 15
    angle  = degree*math.pi/180 

    box1 = get_rotated_coors(box)
    box[-1] += angle 
    box2 = get_rotated_coors(box)
    
    _,giou_coor   = skewiou(box1,box2,'giou',True)
    _,tiou_coor   = skewiou(box1,box2,'tiou',True)
    _,r_giou_coor = skewiou(box1,box2,'r_giou',True)

    coors = [giou_coor,tiou_coor,r_giou_coor]
    label = ['giou','tiou','r_giou']
        
    object_coors = [coor[10:-2] for coor in coors]
    object_coors = [re.split(',| ',i) for i in object_coors]
    for c,ob in enumerate(object_coors):
        ob = [i for i in ob if len(i)!= 0 ]
        ob = ob[:-2]
        ob = [float(i) for i in ob]
        object_coors[c] = ob
    for n,coor in enumerate(object_coors):
        img=cv2.imread(img_path,1)
        img = cv2.polylines(img,[np.array(box1).reshape(-1,2).astype(np.int32)],True,(0,0,255),2)  	
        img = cv2.polylines(img,[np.array(box2).reshape(-1,2).astype(np.int32)],True,(255,0,0),2)
        fig = plt.figure(num=n)
        _img = cv2.polylines(img,[np.array(coor).reshape(-1,2).astype(np.int32)],True,(0,255,0),2)
        plt.imshow(_img)
        plt.title(label[n])
    plt.show()

## 绘制不同iou计算方式的角度和iou的变化曲线
def angle_iou_curve(box):
    iou=[]; giou=[]; tiou=[]; r_giou=[]
    
    degrees = [i for i in range(0,90,1)]
    angles  = [degree*math.pi/180 for degree in degrees]
    for cnt,a in enumerate(angles):
        cx = box[0]; cy = box[1]; w = box[2]; h = box[3]
        xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
        box1 = [xmin, ymax, xmin, ymin, xmax, ymin, xmax, ymax]
        box[-1] = a
        box2 = get_rotated_coors(box)

        _iou = skewiou(box1,box2,'iou')
        _giou = skewiou(box1,box2,'giou')
        _tiou = skewiou(box1,box2,'tiou')
        _r_giou = skewiou(box1,box2,'r_giou')

        iou.append(_iou)
        giou.append(_giou)
        tiou.append(_tiou)
        r_giou.append(_r_giou)
    ious=[iou,giou,tiou,r_giou]
    color = ['C0','C1','C2','red']
    label = ['iou','giou','tiou','r_giou']
    for i,info in enumerate(ious):
        plt.figure(figsize=(10, 8),num='iou-angle') 
        plt.plot(degrees, info, color=color[i], linestyle="-",  linewidth=1, label=label[i]) 
    plt.grid()
    plt.legend()
    plt.show()


## 测试某个iou方式下修正后的效果

# 经过sin(a)的修正使得交集的面积几乎不随着角度而变化（曲线是平的），下一步考虑一个合适的线性函数乘上去即可，即修改lcoef
# 还有暂时没解决后面再补上：
#     - 约束范围归一化：不能保证实际情况下乱七八糟的anchor是否出现奇葩的wha导致coef爆炸从而iou出现bug，必须对输入归一化，并且提供画图api监测
#     - 还要探索angle和wh的变化关系：实际运行时pred_box与gt不一样，看是否全部范围有效
#     - linear系数的拟定
def modified_iou_curve(box):
    w = box[2]; h = box[3]
    critical_angle = math.asin(2*h/w)
    assert critical_angle>0 and critical_angle < 0.5*math.pi ,'emmm,角度超了...'

    iou=[]; giou=[]; tiou=[]; r_giou=[]
    miou=[]; mgiou=[]; mtiou=[]; mr_giou=[]
    degrees = [i for i in range(1,90,1)]
    angles  = np.array([degree*math.pi/180 for degree in degrees])
    if critical_angle:
        mpos = angles>critical_angle
    for cnt,a in enumerate(angles):
        cx = box[0]; cy = box[1]; w = box[2]; h = box[3]
        xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
        box1 = [xmin, ymax, xmin, ymin, xmax, ymin, xmax, ymax]
        box[-1] = a
        box2 = get_rotated_coors(box)

        _iou = skewiou(box1,box2,'iou')
        _giou = skewiou(box1,box2,'giou')
        _tiou = skewiou(box1,box2,'tiou')
        _r_giou = skewiou(box1,box2,'r_giou')

        iou.append(_iou)
        giou.append(_giou)
        tiou.append(_tiou)
        r_giou.append(_r_giou)

        if mpos[cnt]:
            lcoef = 1
            coef = math.sin(a)*lcoef
            _iou = _iou*coef
            _giou = _giou*coef
            _tiou = _tiou*coef
            _r_giou = _r_giou*coef

        miou.append(_iou)
        mgiou.append(_giou)
        mtiou.append(_tiou)
        mr_giou.append(_r_giou)

    ious=[iou,miou]
    color = ['red','C0','C1','C2']
    label = ['iou','miou','giou','r_giou']
    # 全显示
    # ious=[iou,miou,giou,mgiou,r_giou,mr_giou]
    # color = ['red','C0','C1','C2','C3','teal','red','violet']
    # label = ['iou','miou','giou','mgiou','r_giou','mr_giou']
    # 缩放到等同尺度
    s = iou[16]/miou[16]
    for c,i in enumerate(miou):
        if c > 15:
            miou[c] *= s 
    ious=[iou,miou,giou,r_giou]
    for i,info in enumerate(ious):
        plt.figure(figsize=(10, 8),num='iou-angle') 
        plt.plot(degrees, info, color=color[i], linestyle="-",  linewidth=1, label=label[i]) 
    plt.grid()
    plt.legend()
    plt.show()

# 点逆时针旋转(可视化)
# usage:  point_rotate(math.radians(angle),x,y,centerx,centery)
def point_rotate(angle,x,y,centerx,centery,vis=True):
    x = np.array(x)
    y = np.array(y)
    nRotatex = (x-centerx)*math.cos(angle) - (y-centery)*math.sin(angle) + centerx
    nRotatey = (x-centerx)*math.sin(angle) + (y-centery)*math.cos(angle) + centery
    if vis:
        plt.plot([centerx,x],[centery,y])
        plt.plot([centerx,nRotatex],[centery,nRotatey])
        plt.show()
    return nRotatex,nRotatey
    



def rotate_box(corners,angle,  cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
    #
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cx
    # M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    # calculated = calculated.reshape(-1, 8)

    return calculated

if __name__ == "__main__":

    # test
    # a=[2,0,2,2,0,0,0,2]   
    # b=[1,1,4,1,4,4,1,4]
    # a = [ 0.96141, -8.37391, -0.96141, -8.37391, -0.96141,  8.37391,  0.96141,  8.37391]
    # b = [-7.44582, -7.99296, -9.63782, -5.14200,  7.44582,  7.99296,  9.63782,  5.14200]
    # iou=skewiou(a,b,'iou')
    # print(iou)
    # iou2=skewiou(a,b,'tiou')
    # iou3=skewiou(a,b,'giou')

    # box = [834.6619, 442.8605, 681.504, 98.4231, -1.188996]
    # img_path ='/py/datasets/HRSC2016/yolo-dataset/Images/100000631.jpg'

    # vis_unionboxes(box,img_path)
    # angle_iou_curve(box)
    # modified_iou_curve(box)

    import math
    from drawbox import drawbox
    coor = np.array([[285.95 , 239.58]])
    a= -1.5099425238407065
    new_coor = rotate_box(coor, a, 304 ,304 ,608, 608)
    print(new_coor)
    
    x,y = [285.95 , 239.58]
    centerx,centery=(304,304)
    rx,ry=point_rotate(a,x,y,centerx,centery)
    print(rx,ry)





























# https://www.osgeo.cn/shapely/manual.html#polygons


