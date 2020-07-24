from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import  os 
import ipdb
import cv2
import numpy as np
import time

# 代码功能：过分割
# 关注：画图--自动将画的图放到最大显示
# plt.figure(figsize=(10, 10))
# 注意：尺寸的参数是缩小1000倍的，也就是实际上是（1000,1000）的画布；
#	    实际使用根据显示器分辨率调整即可，允许超出。超出也是全屏
# 该方法各种画布都适用（包括subplot），如下例子

def Superpix(img_path):
	plt.figure(figsize=(10, 10))	# 注意这个设置画布大小实现自动填充

	img = io.imread(img_path)
	# ipdb.set_trace()
	start = time.clock()
	segments = slic(img, n_segments=160, compactness=10)
	elapsed = (time.clock() - start)
	print("1 Time used:",elapsed)
	out=mark_boundaries(img,segments,color=(255, 0, 0),mode='subpixel')
	# print(segments)

	plt.subplot(131)
	plt.title("n_segments=160")
	plt.imshow(out)

	start = time.clock()
	segments2 = slic(img, n_segments=300, compactness=10)
	elapsed = (time.clock() - start)
	print("2 Time used:",elapsed)
	out2=mark_boundaries(img,segments2,mode='subpixel')

	plt.subplot(132)
	plt.title("n_segments=300")
	plt.imshow(out2)

	start = time.clock()
	segments3 = slic(img, n_segments=200, compactness=20)
	elapsed = (time.clock() - start)
	print("3 Time used:",elapsed)
	out3=mark_boundaries(img,segments3,mode='subpixel')

	plt.subplot(133)
	plt.title("n_segments=300/100")
	plt.imshow(out3)


	plt.show()




if __name__ == '__main__':

	path = '/py/datasets/bbox_cut/hard_ship'

	
	if os.path.isdir(path):
		files = os.listdir(path)
		img_files = [file[:-4]+'.jpg' for file in files] 
		for img_file in img_files:
			print(img_file)
			img_path = os.path.join(path,img_file)
			Superpix(img_path)
	else:
		Superpix(path)



