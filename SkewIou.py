import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形
from shapely.geometry import Polygon
import random
import math
import matplotlib.pyplot as plt
import cv2

# 不用关注点的顺序，会自动编排，但是建议最好不要交叉，按照顺/逆时针输入
# Polygon.wkt/MultiPoint.wkt to vis
def skewiou(box1, box2,mode='iou'):
    box1=np.asarray(box1)
    box2=np.asarray(box2)
    a=np.array(box1).reshape(4, 2)   
    b=np.array(box2).reshape(4, 2)
    # 所有点的最小凸的表示形式，四边形对象，会自动计算四个点，最后顺序为：左上 左下  右下 右上 左上
    poly1 = Polygon(a).convex_hull  
    poly2 = Polygon(b).convex_hull

    if not poly1.is_valid or not poly2.is_valid:
        print('formatting errors for boxes!!!! ')
        return 0

    inter = Polygon(poly1).intersection(Polygon(poly2)).area
    if   mode == 'iou':
        union = poly1.area + poly2.area - inter
    elif mode =='tiou':
        union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
        union = MultiPoint(union_poly).convex_hull.area
    elif mode == 'giou':
        union_poly = np.concatenate((a,b))   
        union = MultiPoint(union_poly).envelope.area
    elif mode== 'r_giou':
        union_poly = np.concatenate((a,b))   
        union = MultiPoint(union_poly).minimum_rotated_rectangle.area
    else:
        print('incorrect mode!')
    if union == 0:
        return 0
    else:
        return inter/union




# test
# a=[2,0,2,2,0,0,0,2]   
# b=[1,1,4,1,4,4,1,4]


#shaply:https://www.osgeo.cn/shapely/manual.html#polygons
