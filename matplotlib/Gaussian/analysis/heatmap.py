import numpy as np
import matplotlib.pyplot as plt
import json, glob
import scipy.stats
import sys, time, os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from linecache import getline, clearcache
from scipy.integrate import simps
from scipy.constants import *
from scipy.stats import multivariate_normal

def integrate_simps(mesh, func):
    nx, ny = func.shape
    px, py = mesh[0][int(nx/2), :], mesh[1][:, int(ny/2)]
    val = simps( simps(func, px), py )
    return val

def normalize_integrate(mesh, func):
    return func / integrate_simps(mesh, func)

def moment(mesh, func, index):
    ix, iy = index[0], index[1]
    g_func = normalize_integrate(mesh, func)
    fxy = g_func * mesh[0]**ix * mesh[1]**iy
    val = integrate_simps(mesh, fxy)
    return val

def moment_seq(mesh, func, num):
    seq = np.empty([num, num])
    for ix in range(num):
        for iy in range(num):
            seq[ix, iy] = moment(mesh, func, [ix, iy])
    return seq

# 积分得到分布采样点的中心坐标
def get_centroid(mesh, func):
    # mesh为(x, y) pair，func为value
    dx = moment(mesh, func, (1, 0))
    dy = moment(mesh, func, (0, 1))
    return dx, dy

def get_weight(mesh, func, dxy):
    g_mesh = [mesh[0] - dxy[0], mesh[1] - dxy[1]]
    lx = moment(g_mesh, func, (2, 0))
    ly = moment(g_mesh, func, (0, 2))
    return np.sqrt(lx), np.sqrt(ly)

# 根据样本计算协方差矩阵
def get_covariance(mesh, func, dxy):
    g_mesh = [mesh[0] - dxy[0], mesh[1] - dxy[1]]
    Mxx = moment(g_mesh, func, (2, 0))
    Myy = moment(g_mesh, func, (0, 2))
    Mxy = moment(g_mesh, func, (1, 1))
    return np.array([[Mxx, Mxy], [Mxy, Myy]])

def plot_contour_sub(mesh, func, loc=[0, 0], title="name", pngfile="./name"):
    sx, sy = loc
    nx, ny = func.shape
    xs, ys = mesh[0][0, 0], mesh[1][0, 0]
    dx, dy = mesh[0][0, 1] - mesh[0][0, 0], mesh[1][1, 0] - mesh[1][0, 0]
    mx, my = int ( (sy-ys)/dy ), int ( (sx-xs)/dx )
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    ax.set_aspect('equal')
    ax_x = divider.append_axes("bottom", 1.0, pad=0.5, sharex=ax)
    ax_x.plot(mesh[0][mx, :], func[mx, :])
    ax_x.set_title("y = {:.2f}".format(sy))
    ax_y = divider.append_axes("right" , 1.0, pad=0.5, sharey=ax)
    ax_y.plot (func[:, my], mesh[1][:, my])
    ax_y.set_title("x = {:.2f}".format(sx))
    # im = ax.contourf (*mesh, func, cmap="jet", levels =[0,0.1,0.2,0.8,  0.9, 0.95, 1.0 ] )
    im = ax.contourf (*mesh, func, cmap="jet" )
    ax.set_title(title)
    # 右端的色彩条是否添加
    plt.colorbar(im, ax=ax, shrink=0.9)
    plt.savefig(pngfile + ".png")

# mesh是二维离散点构成的定义域面；
# sxy分布中心点u
# rxy是协方差矩阵
def make_gauss(mesh, sxy, rxy, rot):
    # 这里是对自变量的线性变换，将标准正态分布扩展到一般正态分布；和标准公式是等价的
    x, y = mesh[0] - sxy[0], mesh[1] - sxy[1]
    px = x * np.cos(rot) - y * np.sin(rot)
    py = y * np.cos(rot) + x * np.sin(rot)
    fx = np.exp(-0.5 * (px/rxy[0])**2)
    fy = np.exp(-0.5 * (py/rxy[1])**2)
    return fx * fy

# 注意这里的wh是半轴长度
def make_gauss_from_rect(mesh, cx, cy, w, h, theta):
    R = np.array([ np.cos(theta),  -np.sin(theta),
                   np.sin(theta),  np.cos(theta)]).reshape(2,2)
    S = np.array([w, 0, 0, h]).reshape(2,2)
    T = np.dot(R, S) 
    # x, y = mesh[0] , mesh[1] 
    # px = trans[0, 0] * x + trans[0, 1] * y
    # py = trans[1, 0] * x + trans[1, 1] * y
    # fx = np.exp(-0.5 * (px)**2)
    # fy = np.exp(-0.5 * (py)**2)
    # fxy = fx * fy
    mean = (cx, cy)
    
    conv = T * T.T
    return  mean, conv

def get_param_from_rect(cx, cy, w, h, theta):
    mean = (cx, cy)
    conv = 0.25 * np.array([
        w**2 * np.cos(theta)**2 + h**2 * np.sin(theta)**2,
        (w**2 - h**2) * np.sin(theta) * np.cos(theta), 
        (w**2 - h**2) * np.sin(theta) * np.cos(theta), 
        w**2 * np.sin(theta)**2 + h**2 * np.cos(theta)**2,
    ]).reshape(2,2)
    import ipdb; ipdb.set_trace()
    return mean, conv


def Reconst_Gaussian():
    '''
    理解二元高斯分布：
    1. 根据已知参数生成二元高斯分布的数据；
    2. 由数据估计出均值向量和协方差矩阵；
    3. 根据数据重新构造二元高斯分布。
    '''
    nx, ny = 500, 500   # 两个变量的数据取点个数
    lx, ly = 200, 150   # 两变量的定义域
    sx, sy = 50, 10     # 两个一维高斯的均值
    rx, ry = 40, 25     # 两个一维高斯的标准差
    rot    = 30         # 线性映射矩阵

    px = np.linspace(-1, 1, nx) * lx  
    py = np.linspace(-1, 1, ny) * ly
    mesh = np.meshgrid (px, py)     # 铺成xy二元的定义域
    fxy0 = make_gauss(mesh, [sx, sy], [rx, ry], np.deg2rad(rot)) * 10
    
    # s0xy = get_centroid(mesh, fxy0)
    # w0xy = get_weight(mesh, fxy0, s0xy)
    # fxy1 = make_gauss(mesh, s0xy, w0xy, np.deg2rad(0))
    # s1xy = get_centroid(mesh, fxy1)
    # w1xy = get_weight(mesh, fxy1, s1xy)
    
    s0xy = get_centroid(mesh, fxy0)
    w0xy = get_covariance(mesh, fxy0, s0xy)
    # 根据样本估计得到的协方差矩阵和均值构建二元高斯分布
    fxy1 = multivariate_normal.pdf(np.stack(mesh, -1), mean=s0xy, cov=w0xy)
    s1xy = get_centroid(mesh, fxy1)
    w1xy = get_covariance(mesh, fxy1, s1xy)

    print([sx, sy], s0xy, s1xy)
    print([rx, ry], w0xy, w1xy)

    plot_contour_sub(mesh, fxy0, loc=s0xy, title="Gaussian Distribution", pngfile="./fxy0")
    # plot_contour_sub(mesh, fxy1, loc=s1xy, title="Reconst" , pngfile="./fxy1")
    # 按照指定dpi存储
    # plt.savefig('sample.png',dpi=600)
    plt.show()

def Const_from_Rect():
    '''
    1. 定义标准正态分布；
    2. 根据矩形参数进行几何上的映射得到一般性的高斯分布，并获取参数；
    3. 根据参数重构二元高斯分布。
    note: 矩形对应的椭圆只是高斯椭圆族的一部分，需要确定常数c才能得到分布参数
          而其恰好服从卡方分布.
    '''
    nx, ny = 500, 500   # 两个变量的数据取点个数
    lx, ly = 200, 150   # 两变量的定义域
    cx, cy, w, h, theta = 50, 50, 100, 50, 60
    prob = 0.5
    s = scipy.stats.chi2.ppf(prob, 2)

    px = np.linspace(-1, 1, nx) * lx  
    py = np.linspace(-1, 1, ny) * ly
    mesh = np.meshgrid (px, py)  
    # 从矩形参数得到原始分布，以及分布的参数
    # mean, conv = make_gauss_from_rect(mesh, cx, cy, 0.5*w/np.sqrt(s), 0.5*h/np.sqrt(s), np.deg2rad(theta)) 
    mean, conv = get_param_from_rect(cx, cy, w, h, np.deg2rad(theta))
    print(conv)
    # s0xy = get_centroid(mesh, fxy0)
    # 从参数重建高斯
    fxy1 = multivariate_normal.pdf(np.stack(mesh, -1), mean=mean, cov=conv)
    s1xy = get_centroid(mesh, fxy1)
    # plot_contour_sub(mesh, fxy0, loc=s0xy, title="Original", pngfile="./fxy0")
    plot_contour_sub(mesh, fxy1, loc=s1xy, title="Reconst", pngfile="./fxy1")
    plt.show()


if __name__ == "__main__":
    Reconst_Gaussian()
    # Const_from_Rect()

