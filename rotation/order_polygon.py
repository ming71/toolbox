import math
import random
import numpy as np
import matplotlib.pyplot as plt

pts = [[1,1],[0,2],[2,1],[2,2],[2,0],[0,1],[1,0],[0,0]]

def carttopolar(x,y,x0=0,y0=0):
    '''
    cartisian to polar coordinate system with origin shift to x0,y0
    '''
    x1=x-x0
    y1=y-y0
    #print('(x0,y0)sort',x0,y0)
    r = np.sqrt(x1**2+y1**2)
    t = np.arctan2(y1,x1)*180/math.pi
    if y1<0:
        t=360+t
    #print('x,y,r,t',x,y,r,t)
    return r,t

def sort_aniclkwise(xy_list,x0=None,y0=None):
    '''
    Sort points anit clockwise with x0 y0 as origin
    '''
    if x0 is None and y0 is None:
        (x0,y0)=np.mean(xy_list,axis=0).tolist()
    elif x0 is None:
        (x0,_)=np.mean(xy_list,axis=0).tolist()
    elif y0 is None:
        (_,y0)=np.mean(xy_list,axis=0).tolist()
    print('origin used:',[x0,y0])  

    for i in range(len(xy_list)):
          xy_list[i].append(i)

    xy_list1 = sorted(xy_list, key=lambda a_entry: carttopolar(a_entry[0], a_entry[1],x0,y0)[1])

    sort_index = []  
    for x in xy_list1:
          sort_index.append(x[2])
          del x[2]


    return xy_list1, sort_index

def draw(pts):
    color_dict = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    idx = random.randint(0,7)
    pts = np.array(pts).reshape(-1,2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    plt.scatter(xs,ys)
    plt.plot(xs,ys, color=color_dict[idx])
draw(pts)
# spts,_=sort_aniclkwise(pts)
# draw(spts)
plt.show()

# import ipdb; ipdb.set_trace()
