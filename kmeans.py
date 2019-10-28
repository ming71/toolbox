'''
2019.10.8  ming71
功能:  对box进行anchor的kmeans聚类
注意:  
    - 停止条件是最小值索引不变而不是最小值不变，会造成早停，可以改
    - 暂时仅支持voc标注,如需改动再重写get_all_boxes函数即可
评价方法：
    - anchor聚类采用iou评价 / 可视化(method1情况下)
    - area和ratio聚类采用可视化散点图
'''
import numpy as np
import glob
import os
import cv2
from decimal import Decimal
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



## a sample for kmeans via sklearn:
    # import numpy as np
    # from sklearn.cluster import KMeans
    # data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
    # import ipdb; ipdb.set_trace()
    # #假如我要构造一个聚类数为3的聚类器
    # estimator = KMeans(n_clusters=3)#构造聚类器
    # estimator.fit(data)#聚类
    # label_pred = estimator.labels_ #获取聚类标签
    # centroids = estimator.cluster_centers_ #获取聚类中心
    # inertia = estimator.inertia_ # 获取聚类准则的总和

class Kmeans:
    def __init__(self, cluster_number, all_boxes, save_path=None):
        self.cluster_number = cluster_number
        self.all_boxes = all_boxes
        self.save_path = save_path

   # 输入两个二维数组:所有box和种子点box
   # 输出[num_boxes, k]的结果
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]          
        k = self.cluster_number     #类别

        box_area = boxes[:, 0] * boxes[:, 1]    #列表切片操作：取所有行0列和1列相乘 ，得到gt的面积的行向量
        box_area = box_area.repeat(k)           #行向量进行重复
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]      #种子点的面积行向量
        cluster_area = np.tile(cluster_area, [1, n])        
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area + 1e-16)
        assert (result>0).all() == True , 'negtive anchors present , cluster again!'
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy
    
    def result2txt(self, data):     
        f = open(self.save_path, 'w')      
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()                              #最终输出的是w1,h1,w2,h2,w3,h3,...


    def anchor_clusters(self):
        boxes = np.array(self.all_boxes)                                #返回全部gt的宽高二维数组
        k=self.cluster_number
        ############   K-means聚类计算  ######

        #####  Method 1 : sklearn implemention 
        estimator = KMeans(n_clusters=k)
        estimator.fit(boxes)             #聚类
        label_pred = estimator.labels_   #获取聚类标签
        centroids = estimator.cluster_centers_ #获取聚类中心
        centroids = np.array(centroids)
        result = centroids[np.lexsort(centroids.T[0, None])]              #将得到的三个anchor按照宽进行从小到大，重新排序
        print("K anchors:\n {}\n".format(result))
        print("Accuracy: {:.2f}%\n".format(self.avg_iou(boxes, result) * 100))
        plt.figure()
        plt.scatter(boxes[:,0], boxes[:,1], marker='.',c=label_pred)
        plt.xlabel('anchor_w')
        plt.ylabel('anchor_h')
        plt.title('anchor_clusters')
        for c in centroids:
            plt.annotate(s='cluster' ,xy=c ,xytext=c-20,arrowprops=dict(facecolor='red',width=3,headwidth = 6))
            plt.scatter(c[0], c[1], marker='*',c='red',s=100)

        #####  Method 2 : 自己写一边kmeans，这个的iou更高，推荐使用
        # #注意：这里代码选择的停止聚类的条件是最小值的索引不变，而不是种子点的数值不变。这样的误差理论会大一点，其实关系不大。
        # box_number = boxes.shape[0]                 # box个数
        # distances = np.empty((box_number, k))       # 初始化[box_number , k]二维数组，存放自定义iou距离（obj*anchor）
        # last_nearest = np.zeros((box_number,))       # [box_number , ]的标量
        # np.random.seed()                           
        # clusters = boxes[np.random.choice(
        #     box_number, k, replace=False)]  # 种子点随机初始化

        # # 种子点一旦重复会有计算错误,避免!   
        # while True :
        #     uniques_clusters = np.unique(clusters,axis=0)
        #     if  len(uniques_clusters)==len(clusters) :
        #         break
        #     clusters = boxes[np.random.choice(box_number, k, replace=False)]
        
        # # k-means
        # while True:
        #     # 每轮循环，计算种子点外所有点各自到k个种子点的自定义距离，并且按照距离各个点找离自己最近的种子点进行归类；计算新的各类中心；然后下一轮循环
        #     distances = 1 - self.iou(boxes, clusters)   # iou越大,距离越小

        #     current_nearest = np.argmin(distances, axis=1)  # 展开为box_number长度向量,代表每个box当前属于哪个种子点类别(0,k-1) 
        #     if (last_nearest == current_nearest).all():     # 每个box的当前类别所属和上一次相同,不再移动聚类
        #         break                                       

        #     #计算新的k个种子点坐标
        #     for cluster in range(k):
        #         clusters[cluster] = np.median(boxes[current_nearest == cluster], axis=0) # 只对还需要聚类的种子点进行位移
        #     last_nearest = current_nearest
        # result = clusters[np.lexsort(clusters.T[0, None])]              #将得到的三个anchor按照宽进行从小到大，重新排序
        # print('\n-----anchor_cluster-----\n')
        # print("K anchors:\n {}\n".format(result))
        # print("Accuracy: {:.2f}%\n".format(self.avg_iou(boxes, result) * 100))
        
        # if self.save_path:
        #     self.result2txt(result)
            
        #     ## 聚类结果分析
        #     with open(self.save_path,'r') as f:
        #         contents = f.read()
        #         w = list(map(int, contents.split(',')[::2]))
        #         h = list(map(int, contents.split(',')[1::2]))
        #         anchors = [anchor for anchor in zip(w,h)]
        #         ratio = [Decimal(anchor[0]/anchor[1]).quantize(Decimal('0.00')) for anchor in anchors] 
        #         ratio.sort()
        #         area =  [Decimal(anchor[0]*anchor[1]).quantize(Decimal('0.00')) for anchor in anchors] 
        #         area.sort()
        #         #####   自定义需要分析的数据  ###
        #         squre=[float(s)**0.5 for s in area]
        #         print('ratio:\n{}\n\narea:\n{}\n'.format(ratio,area))
        #         print('sqrt(area):\n{}'.format(squre))

    ## 懒得重写，直接用sklearn
    def area_cluster(self,vis=False):
        # 面积聚类
        boxes = np.array(self.all_boxes)
        areas = boxes[:,0]*boxes[:,1]

        estimator = KMeans(n_clusters=self.cluster_number)
        estimator.fit(areas.reshape(-1,1))             #聚类
        label_pred = estimator.labels_          #获取聚类标签
        centroids = estimator.cluster_centers_  #获取聚类中心
        centroids = centroids[np.lexsort(centroids.T)]  # 排个序
        centroids = np.array([int(i) for i in centroids]).reshape(-1,1) # 取个整
        print('\n-----area_cluster-----\n')
        print(centroids)  
        if vis:
            plt.figure()
            plt.scatter(range(len(areas)), areas.squeeze(), marker='.',c=label_pred)
            plt.xlabel('gt_num')
            plt.ylabel('area')
            plt.title('area_cluster')
            for c in centroids:
                xy = np.array([int(0.5*len(boxes)),c.item()])
                plt.scatter(int(0.5*len(boxes)),c.item(), marker='*',c='red',s=100)


    def ratio_cluster(self,vis=False):
        # 宽高比聚类
        boxes = np.array(self.all_boxes)
        ratios = boxes[:,0]/boxes[:,1]

        estimator = KMeans(n_clusters=self.cluster_number)
        estimator.fit(ratios.reshape(-1,1))             #聚类
        label_pred = estimator.labels_          #获取聚类标签
        centroids = estimator.cluster_centers_  #获取聚类中心
        centroids = centroids[np.lexsort(centroids.T)]  # 排个序(从小到大)
        # 表示为分子或分母1便于直观观察
        print('\n-----ratio_cluster-----\n')
        for i,c in enumerate(centroids):
            num, den = c.item().as_integer_ratio()
            if c > 1 : num /= den ; den = 1 ; num = Decimal(num).quantize(Decimal('0.00'))
            if c < 1 : den /= num ; num = 1 ; den = Decimal(den).quantize(Decimal('0.00'))
            ratio = str(num) + '/' +str(den)
            print(ratio)
        if vis:
            plt.figure()
            plt.scatter(range(len(ratios)), ratios.squeeze(), marker='.',c=label_pred)
            plt.xlabel('gt_num')
            plt.ylabel('ratio')
            plt.title('ratio_cluster')
            for c in centroids:
                xy = np.array([int(0.5*len(boxes)),c.item()])
                plt.scatter(int(0.5*len(boxes)),c.item(), marker='*',c='red',s=100)


# 返回所有label的box,形式为[[w1,h1],[w2,h2],...]
def get_all_boxes(path,mode=None):
    assert not mode is None,'Input correct label mode,such as : voc, hrsc, yolo'
    boxes = []

    if mode == 'voc':
        labels = sorted(glob.glob(os.path.join(path, '*.*')))
        for label in labels:
            with open(label,'r') as f:
                contents = f.read()
                objects = contents.split('<object>')
                objects.pop(0)
                if len(objects) == 0: pass

                for object in objects:
                    xmin = int(float(object[object.find('<xmin>')+6 : object.find('</xmin>')]))
                    xmax = int(float(object[object.find('<xmax>')+6 : object.find('</xmax>')]))
                    ymin = int(float(object[object.find('<ymin>')+6 : object.find('</ymin>')]))
                    ymax = int(float(object[object.find('<ymax>')+6 : object.find('</ymax>')]))
                    box_w = xmax - xmin 
                    box_h = ymax - ymin
                    boxes.append((box_w,box_h))
    
    elif mode == 'hrsc': # xml格式
        rotate = True
        labels = sorted(glob.glob(os.path.join(path, '*.*')))
        for label in labels:
            with open(label,'r') as f:
                contents = f.read()
                objects = contents.split('<HRSC_Object>')
                objects.pop(0)
                if len(objects) == 0: pass

                for object in objects:
                    if not rotate:
                        xmin = int(object[object.find('<box_xmin>')+10 : object.find('</box_xmin>')])
                        ymin = int(object[object.find('<box_ymin>')+10 : object.find('</box_ymin>')])
                        xmax = int(object[object.find('<box_xmax>')+10 : object.find('</box_xmax>')])
                        ymax = int(object[object.find('<box_ymax>')+10 : object.find('</box_ymax>')])
                        box_w = xmax - xmin; box_h = ymax - ymin
                    else:   # 旋转框
                        box_w  = int(float(object[object.find('<mbox_w>')+8 : object.find('</mbox_w>')]))
                        box_h  = int(float(object[object.find('<mbox_h>')+8 : object.find('</mbox_h>')]))
                    boxes.append((box_w,box_h))
    
    elif mode == 'yolo':
        labels = sorted(glob.glob(os.path.join(path, '*.txt*')))
        for label in tqdm(labels,desc='Loading labels'):
            img_path = os.path.join(os.path.split(label)[0], os.path.split(label)[1][:-4]+'.jpg')
            height,width,_ = cv2.imread(img_path).shape
            with open(label,'r') as f:
                contents=f.read()
                lines=contents.split('\n')
                lines = [x for x in contents.split('\n')  if x]	 # 移除空格

                for object in lines:
                    coors = object.split(' ')
                    box_w = int(float(coors[3])*width)
                    box_h = int(float(coors[4])*height)
                    boxes.append((box_w,box_h))
    else:
        print('Unrecognized label mode!!')
    return boxes



    
if __name__ == "__main__":
    cluster_number = 3              # 种子点个数
    label_path = '/py/datasets/HRSC2016/yolo-dataset/train'
    save_path  = 'anchor-cluster.txt'   

    all_boxes = get_all_boxes(label_path,'yolo')   
    kmeans = Kmeans(cluster_number, all_boxes, save_path=save_path)
    
    vis = True
    kmeans.anchor_clusters()
    kmeans.area_cluster(vis=vis)
    kmeans.ratio_cluster(vis=vis)
    if vis: 
        plt.show()



'''
K anchors:
 [[153.87419355  31.30967742]
 [343.26412214  52.12366412]
 [574.57024793 109.7231405 ]]

Accuracy: 69.19%


-----area_cluster-----

[[ 12130]
 [ 42951]
 [113378]]

-----ratio_cluster-----

4.18/1
6.50/1
8.75/1

'''