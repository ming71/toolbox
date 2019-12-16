import cv2
import numpy as np
import ipdb 
import matplotlib.pyplot as plt
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import math

from decimal import Decimal

# img1 = cv2.imread('/py/pic/dog1.jpg',0)
# img1 = cv2.Canny(img1, 100, 150)
# img2 = cv2.imread('/py/pic/dog.jpg',0)



# (h,w) = img1.shape[:2]
# center = (w / 2,h / 2)
# M = cv2.getRotationMatrix2D(center,0,3.11)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
# img2 = cv2.warpAffine(img1,M,(w,h))

# cv2.imshow("pic",img1)
# cv2.waitKey(0)

# ----------------- 打印观察规律 ------------------------
# 输入：单个图片路径
# 输出：旋转，缩放，平移对图像hu矩的影响（裁剪参见3D点云更有意义）
def single_compare(path,mode=None):
	dists_r = []
	dists_s = []
	dists_t_h = []
	dists_t_v = []
	dists_c = []
	if mode == 'canny':
		img1 = cv2.imread(path,0)
		img1 = cv2.Canny(img1, 100, 150)
	else:
		img1 = cv2.imread(path,0)
	(h,w) = img1.shape[:2]
	center = (w / 2,h / 2)
# --------------------- 旋转：0-359度 ---------------------
	angle_range = 360
	angle_interval = 1

	x_angle   = [i for i in range(int(angle_range/angle_interval))]

	for i in range(int(angle_range/angle_interval)):
		M = cv2.getRotationMatrix2D(center,i*angle_interval,1.0)
		img2 = cv2.warpAffine(img1,M,(w,h))
		moments1 = cv2.moments(img1)
		humoments1 = cv2.HuMoments(moments1)
		humoments1 = np.log(1e-16+np.abs(humoments1)) # 同样建议取对数
		moments2 = cv2.moments(img2)
		humoments2 = cv2.HuMoments(moments2)
		humoments2 = np.log(1e-16+np.abs(humoments2)) # 同样建议取对数
		dist_r = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
		dists_r.append(dist_r)
		# cv2.imshow("pic",img2)		
		# cv2.waitKey(0)

# ---------------------- 缩放：0.05-10 ----------------------
	scale_range = (0.05,10.0)
	scale_interval = 0.01

	x_scale   = [scale_range[0]+j*scale_interval for j in range(int((scale_range[1]-scale_range[0])/scale_interval))]

	for j in range(int((scale_range[1]-scale_range[0])/scale_interval)):
		M = cv2.getRotationMatrix2D(center,0,scale_range[0]+j*scale_interval)
		img2 = cv2.warpAffine(img1,M,(w,h))
		moments1 = cv2.moments(img1)
		humoments1 = cv2.HuMoments(moments1)
		humoments1 = np.log(1e-16+np.abs(humoments1)) 
		moments2 = cv2.moments(img2)
		humoments2 = cv2.HuMoments(moments2)
		humoments2 = np.log(1e-16+np.abs(humoments2)) 
		dist_s = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
		dists_s.append(dist_s)
		# cv2.imshow("pic",img2)		
		# cv2.waitKey(0)

#  ------------------ 平移（水平方向和垂直方向）:宽高的0.1-0.9，中间0.01为间隔----------------------
	trans_range = (0.1,0.9)
	trans_interval = 0.01

	x_trans = [-trans_range[1]+trans_interval * i for i in range(2*int((1-trans_range[0])/trans_interval))]

	for i in range(2*int((1-trans_range[0])/trans_interval)):
		bias_h = (-trans_range[1]+trans_interval * i) * w 
		bias_v = (-trans_range[1]+trans_interval * i) * h 
		M_h = np.float32([[1,0,bias_h],[0,1,0]]) 	# 平移矩阵M：[[1,0,x],[0,1,y]]
		M_v = np.float32([[1,0,0],[0,1,bias_v]]) 	# 平移矩阵M：[[1,0,x],[0,1,y]]
		img2 = cv2.warpAffine(img1,M_h,(w,h))
		img3 = cv2.warpAffine(img1,M_v,(w,h))
		moments1 = cv2.moments(img1)
		humoments1 = cv2.HuMoments(moments1)
		humoments1 = np.log(1e-16+np.abs(humoments1)) 
		moments2 = cv2.moments(img2)
		humoments2 = cv2.HuMoments(moments2)
		humoments2 = np.log(1e-16+np.abs(humoments2)) 
		moments3 = cv2.moments(img3)
		humoments3 = cv2.HuMoments(moments3)
		humoments3 = np.log(1e-16+np.abs(humoments3)) 
		dist_t_h = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
		dist_t_v = np.linalg.norm(humoments1.reshape(7,) - humoments3.reshape(7,))
		dists_t_h.append(dist_t_h)
		dists_t_v.append(dist_t_v)
		# cv2.imshow("pic",img2)		
		# cv2.waitKey(0)


# ---------------  制图部分 ----------------------------
	plt.figure(1) 
	plt.subplot(2, 2, 1) 
	plt.plot(x_angle, dists_r, color="r", linestyle="-",  linewidth=1) 
	plt.grid()
	plt.xlabel("x_angle")
	plt.ylabel("rotate")

	plt.figure(1) 
	plt.subplot(2, 2, 2) 
	plt.plot(x_scale, dists_s, color="b", linestyle="-", linewidth=1)
	plt.xlabel("x_scale")
	plt.ylabel("scale")
	plt.grid()

	plt.figure(1) 
	plt.subplot(2, 2, 3) 
	plt.plot(x_trans, dists_t_h, color="g", linestyle="-",  linewidth=1) 
	plt.grid()
	plt.xlabel("x_trans_h")
	plt.ylabel("trans_h")

	plt.figure(1) 
	plt.subplot(2, 2, 4) 
	plt.plot(x_trans, dists_t_v, color="g", linestyle="-",  linewidth=1) 
	plt.grid()
	plt.xlabel("x_trans_v")
	plt.ylabel("trans_v")

	plt.show()



# ---------------画三维图同时观察两个变量的影响------------
# 输入可以是一张图，也可以是一组xy下的多个对照z
# 其中z和ylabel是list嵌套list，用于记录多组数据和纵轴名字
def draw_3D_point(x1,x2,y,x1label='x1',x2label='x2',ylabel='dist'):
	# ipdb.set_trace()
	assert len(np.array(x1).shape) == 1 ,'暂时只支持统一的xy进行对比，否则直接多次画图就行'

	fig = plt.figure()	
	x1 , x2 = np.meshgrid(x1 , x2)   # 注意3D图的画法！需要构造这样的xy矩阵形式，不能是向量

	# 只画一张图，对于同样的xy只传入一组z
	if len(np.array(y).shape) == 1:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(x1, x2, y,marker='.',linewidth=0.1)
		 
		ax.set_xlabel(x1label)
		ax.set_ylabel(x2label)
		ax.set_zlabel(ylabel)
	# 传入多组进行对照，画到一张图上去
	else :
		num_fig = len(np.array(y))
		# 大于3张图时，布局为两行；2-3张图仍为一行
		row = 2 if num_fig > 3 else 1
		col = math.ceil(num_fig/2)  if num_fig > 3 else num_fig
		# count:0-7索引y的元素用  p: 1-8 添加画布用
		for count,p in enumerate(range(1,num_fig+1)):
			# 注意申请动态变量以便于将图画到一张上去
			exec ("ax{} = fig.add_subplot(row,col,p ,projection='3d')".format(p))   
			exec("ax{}.scatter(x1, x2, y[count],marker='.',linewidth=0.1)".format(p))
			exec('ax{}.set_xlabel(x1label)'.format(p))
			exec('ax{}.set_ylabel(x2label)'.format(p))
			exec('ax{}.set_zlabel("{}")'.format(p,ylabel[count]))	# 注意这个引号的调用方法！
	# plt.show()



	 

def multiple_compare(path,mode=None,item=['crop','rotate','trans']):
	# 包含旋转+缩放，平移，裁剪的对比
	dists_rs = []
	dists_t_v = []
	dists_t_h = []
	dists_c = []

	if mode == 'canny':
		img1 = cv2.imread(path,0)
		img1 = cv2.Canny(img1, 100, 150)
	else:
		img1 = cv2.imread(path,0)
	(h,w) = img1.shape[:2]
	center = (w / 2,h / 2)

# --------------------  旋转缩放 ----------------------------------------
	'''
		rotate: 0-359,间隔5
		scale:  0.05-5,间隔0.1
	'''
	angle_range = 360
	angle_interval = 5
	scale_range = (0.05,5.0)
	scale_interval = 0.1

	if 'rotate' in item :
		for i in range(int(angle_range/angle_interval)):
			print('\rrotate & scale :{:.2f}%  '.format(i*100/(angle_range/angle_interval)),end='')
			for j in range(int((scale_range[1]-scale_range[0])/scale_interval)):
				M = cv2.getRotationMatrix2D(center,angle_interval*i,scale_range[0]+j*scale_interval)
				img2 = cv2.warpAffine(img1,M,(w,h))
				moments1 = cv2.moments(img1)
				humoments1 = cv2.HuMoments(moments1)
				humoments1 = np.log(1e-16+np.abs(humoments1)) 
				moments2 = cv2.moments(img2)
				humoments2 = cv2.HuMoments(moments2)
				humoments2 = np.log(1e-16+np.abs(humoments2)) 
				dist_r = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
				dists_rs.append(dist_r)
				# cv2.imshow("pic",img2)		
				# cv2.waitKey(0)
		x_angle   = [angle_interval*i for i in range(int(angle_range/angle_interval))]
		x_scale   = [scale_range[0]+j*scale_interval for j in range(int((scale_range[1]-scale_range[0])/scale_interval))]
		draw_3D_point(x_angle,x_scale,dists_rs,x1label='Rotate',x2label='Scale',ylabel='dist')

# -------------------------- 裁剪  ----------------------------------------
	'''
	裁剪(水平和竖直)：宽高为范围，其0.05作为slide步长
	patch 大小设置8个档次：原图短边的0.1-0.8(正方形)
	'''

	patch_range = (0.1,1)
	patch_interval = 0.1 
	stride = 0.05

	if 'crop' in item :
		label_list = []		# 存放打包的label名字
		for c in range(int((patch_range[1]-patch_range[0])/patch_interval)+1):
			p = patch_range[0]+patch_interval*c
			dists_c_patch = []
			patch_size = min(h,w) * p * patch_interval
			for y in range(int(1/stride)):
				print('\rcrop :{:.2f}%  '.format(y*100/((1/stride)*8)),end='')
				for x in range(int(1/stride)): 	
					y0 = int(y * 0.01 * h)
					x0 = int(x * 0.01 * w)
					y1 = int(min(h,patch_size+y0))
					x1 = int(min(w,patch_size+x0))
					img2 = img1[y0:y1 , x0:x1] 
					moments1 = cv2.moments(img1)
					humoments1 = cv2.HuMoments(moments1)
					humoments1 = np.log(1e-16+np.abs(humoments1)) 
					moments2 = cv2.moments(img2)
					humoments2 = cv2.HuMoments(moments2)
					humoments2 = np.log(1e-16+np.abs(humoments2)) 
					dist_c = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
					dists_c_patch.append(dist_c)
			dists_c.append(dists_c_patch)	# 一张图画多张小图，需要打包数据
			# 打包label名，由于python计算机制导致float很长，取固定位小数
			label_list.append('dist( patchsize: {} )'.format(Decimal(p).quantize(Decimal('0.0'))))
		x_pos = [i*0.05 for i in range(int(1/stride))] 
		y_pos = [i*0.05 for i in range(int(1/stride))]	
		
		draw_3D_point(x_pos,y_pos,dists_c,x1label='x1',x2label='x2',ylabel=label_list)


# -------------------------- 平移  ----------------------------------------	
	'''
	水平和垂直四个方向的平移，两种策略：固定x还是固定y滑动
	'''
	trans_range = (0.05,0.95)
	trans_interval = 0.05

	x_trans = [-trans_range[1]+trans_interval * i for i in range(2*int((1-trans_range[0])/trans_interval))]

	for i in range(2*int((1-trans_range[0])/trans_interval)):
		bias_h = (-trans_range[1]+trans_interval * i) * w 
		print('\rtrans:{:.2f}%'.format(i*100/((trans_range[1]-trans_range[0])/trans_interval*2),end=''))
		for j in range(2*int((1-trans_range[0])/trans_interval)):
			bias_v = (-trans_range[1]+trans_interval * j) * h 
			M = np.float32([[1,0,bias_h],[0,1,bias_v]]) 	# 平移矩阵M：[[1,0,x],[0,1,y]]
			img2 = cv2.warpAffine(img1,M,(w,h))
			moments1 = cv2.moments(img1)
			humoments1 = cv2.HuMoments(moments1)
			humoments1 = np.log(1e-16+np.abs(humoments1)) 
			moments2 = cv2.moments(img2)
			humoments2 = cv2.HuMoments(moments2)
			humoments2 = np.log(1e-16+np.abs(humoments2)) 
			dist_t_v = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
			dists_t_v.append(dist_t_v)
			# cv2.imshow("pic",img2)		
			# cv2.waitKey(0)

	for i in range(2*int((1-trans_range[0])/trans_interval)):
		bias_h = (-trans_range[1]+trans_interval * i) * w 
		for j in range(2*int((1-trans_range[0])/trans_interval)):
			bias_v = (-trans_range[1]+trans_interval * j) * h 

			M = np.float32([[1,0,bias_v],[0,1,bias_h]]) 	# 平移矩阵M：[[1,0,x],[0,1,y]]
			img2 = cv2.warpAffine(img1,M,(w,h))
			moments1 = cv2.moments(img1)
			humoments1 = cv2.HuMoments(moments1)
			humoments1 = np.log(1e-16+np.abs(humoments1)) 
			moments2 = cv2.moments(img2)
			humoments2 = cv2.HuMoments(moments2)
			humoments2 = np.log(1e-16+np.abs(humoments2)) 
			dist_t_h = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
			dists_t_h.append(dist_t_h)
			# cv2.imshow("pic",img2)		
			# cv2.waitKey(0)
	x = [ i for i in range(2*int((1-trans_range[0])/trans_interval))]
	y = [ i for i in range(2*int((1-trans_range[0])/trans_interval))]
	label_list = ['hori','vert']
	data_list =[]
	data_list.append(dists_t_h)
	data_list.append(dists_t_v)

	draw_3D_point(x,y,data_list,x1label='x1',x2label='x2',ylabel=label_list)


	plt.show()	





# ----------------计算相似度------------------------
# def cal_similarity():
# ret, thresh = cv2.threshold(img1, 127, 255,0)
# ret, thresh2 = cv2.threshold(img2, 127, 255,0)
# contours,hierarchy = cv2.findContours(thresh,2,1)
# cnt1 = contours[0]
# contours,hierarchy = cv2.findContours(thresh2,2,1)
# cnt2 = contours[0]
# ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
# # ret = np.log(np.abs(ret+1e-16)) 
# print (ret)


# -------计算Hu矩--------------------------------
# moments1 = cv2.moments(img1)
# humoments1 = cv2.HuMoments(moments1)
# humoments1 = np.log(np.abs(humoments1)) # 同样建议取对数

# moments2 = cv2.moments(img2)
# humoments2 = cv2.HuMoments(moments2)
# humoments2 = np.log(np.abs(humoments2)) # 同样建议取对数
# dist = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
# print(dist)

# ---------------统计任意两张图的不变矩直方图----------
def draw_hist(path,mode=None,counter=1):
	files = os.listdir(path)
	dists = []
	cnt=0 
	for i in range(counter):
		print('\r当前进度:{:.2f}%'.format(i*100/counter),end='')
		file = random.sample(files, 2)
		file = [i[:-4]+'.jpg' for i in file]
		# ipdb.set_trace()
		if mode == 'canny':
			img1 = cv2.imread(path+file[0],0)
			img2 = cv2.imread(path+file[1],0)
			img1 = cv2.Canny(img1, 100, 150)
			img2 = cv2.Canny(img2, 100, 150)
		else:
			img1 = cv2.imread(path,0)
			img1 = cv2.Canny(img1, 100, 150)

		# cv2.imshow("pic",img1)
		# cv2.waitKey(0)
		moments1 = cv2.moments(img1)
		humoments1 = cv2.HuMoments(moments1)
		humoments1 = np.log(1e-16+np.abs(humoments1)) # 同样建议取对数

		moments2 = cv2.moments(img2)
		humoments2 = cv2.HuMoments(moments2)
		humoments2 = np.log(1e-16+np.abs(humoments2)) # 同样建议取对数
		dist = np.linalg.norm(humoments1.reshape(7,) - humoments2.reshape(7,))
		dists.append(dist)
		if dist>10:
			cnt+=1
	x = [i for i in range(counter)]
	print('dist>10占比：{}%'.format(100*cnt/counter))

	plt.hist(dists, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
	plt.xlabel("interval")
	plt.ylabel("frequency")
	plt.title("hist")
	plt.show()


def main():
	path = '/py/pic/dog1.jpg'
	folder_path = '/py/yolov3/data/datasets/VOC2007/JPEGImages/'
	mode = 'canny'
	item = [ 'trans' ]   #  'crop' , 'rotate' , 'trans'

	# draw_hist(folder_path,mode = mode,counter=1000)

	# single_compare(path,mode = mode)
	multiple_compare(path,mode = 'canny',item=item)


if __name__ == "__main__":
	main()