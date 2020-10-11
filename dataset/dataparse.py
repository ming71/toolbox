import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  

import glob
from decimal import Decimal
from sklearn.cluster import KMeans


def parse(labels, out_dir):
    # 统计量
    classes = {}
    widths = []
    heights = []
    ratios = [] # w/h
    thetas = [] # x+轴逆时针旋转碰到的第一条边为角度theta，逆时针为负
    areas = []

    files = os.listdir(labels)
    pbar = tqdm(files)
    for filename in pbar:
        pbar.set_description("dataset parsing")
        with open(os.path.join(labels, filename), 'r') as f:
            objs = f.readlines()
            for obj in objs:
                classname, x1, y1, x2, y2, x3, y3, x4, y4 = obj.strip().split()
                x1, y1, x2, y2, x3, y3, x4, y4 = [x for x in map(eval, [x1, y1, x2, y2, x3, y3, x4, y4])]
                ((cx,cy), (w,h), theta) = cv2.minAreaRect(np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32).reshape(4,2)) 
                if classname in classes.keys():
                    classes[classname] += 1
                else:
                    classes[classname] = 1
                widths.append(w)
                heights.append(h)
                areas.append(w*h)
                ratios.append(max(w,h)/min(w,h))
                thetas.append(theta)
    # 输出统计结果
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'num_per_class.txt'), 'w') as fc:
        cls_names = '\n'.join(classes.keys())
        num_per_class = str(classes).replace(',','\n')
        s = cls_names + '\n\n' + num_per_class
        fc.write(s)
    xtick = np.arange(len(widths))
    plt.scatter(xtick, np.array(ratios), s=10)
    plt.savefig(os.path.join(out_dir, 'ratios.png'))
    plt.clf()
    plt.scatter(xtick, np.array(thetas), s=10)
    plt.savefig(os.path.join(out_dir, 'thetas.png'))
    plt.clf()
    plt.scatter(xtick, np.array(areas), s=10)
    plt.savefig(os.path.join(out_dir, 'areas.png'))
    plt.clf()

    all_boxes = [x for x in zip(widths,heights)]
    return all_boxes


class Kmeans:
    def __init__(self, cluster_number, all_boxes, save_path=None):
        self.cluster_number = cluster_number
        self.all_boxes = all_boxes
        self.save_path = save_path
        self.cluster_res = ''

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
    

    def anchor_clusters(self):
        boxes = np.array(self.all_boxes)                                #返回全部gt的宽高二维数组
        k=self.cluster_number
        estimator = KMeans(n_clusters=k)
        estimator.fit(boxes)             #聚类
        label_pred = estimator.labels_   #获取聚类标签
        centroids = estimator.cluster_centers_ #获取聚类中心
        centroids = np.array(centroids)
        result = centroids[np.lexsort(centroids.T[0, None])]              #将得到的三个anchor按照宽进行从小到大，重新排序
        # print("K anchors:\n {}\n".format(result))
        # print("Accuracy: {:.2f}%\n".format(self.avg_iou(boxes, result) * 100))
        plt.figure()
        plt.scatter(boxes[:,0], boxes[:,1], marker='.',c=label_pred)
        plt.xlabel('anchor_w')
        plt.ylabel('anchor_h')
        plt.title('anchor_clusters')
        for c in centroids:
            plt.annotate(s='cluster' ,xy=c ,xytext=c-20,arrowprops=dict(facecolor='red',width=3,headwidth = 6))
            plt.scatter(c[0], c[1], marker='*',c='red',s=100)
        plt.savefig(os.path.join(self.save_path, 'anchor_clusters.png'))
        plt.clf()
        self.cluster_res = self.cluster_res + 'K anchors:\n' + str(result) + '\n\n--------------\n\n'
    

    def area_cluster(self):
        boxes = np.array(self.all_boxes)
        areas = boxes[:,0]*boxes[:,1]

        estimator = KMeans(n_clusters=self.cluster_number)
        estimator.fit(areas.reshape(-1,1))             #聚类
        label_pred = estimator.labels_          #获取聚类标签
        centroids = estimator.cluster_centers_  #获取聚类中心
        centroids = centroids[np.lexsort(centroids.T)]  # 排个序
        centroids = np.array([int(i) for i in centroids]).reshape(-1,1) # 取个整
        # print('\n-----area_cluster-----\n')
        # print(centroids)  
        self.cluster_res = self.cluster_res + 'area_cluster:\n'+ str(centroids) + '\n\n--------------\n\n'
        plt.figure()
        plt.scatter(range(len(areas)), areas.squeeze(), marker='.',c=label_pred)
        plt.xlabel('gt_num')
        plt.ylabel('area')
        plt.title('area_cluster')
        for c in centroids:
            xy = np.array([int(0.5*len(boxes)),c.item()])
            plt.scatter(int(0.5*len(boxes)),c.item(), marker='*',c='red',s=100)
        plt.savefig(os.path.join(self.save_path, 'area_cluster.png'))
        plt.clf()

    def ratio_cluster(self):
        boxes = np.array(self.all_boxes)
        ratios = boxes[:,0]/boxes[:,1]

        estimator = KMeans(n_clusters=self.cluster_number)
        estimator.fit(ratios.reshape(-1,1))             #聚类
        label_pred = estimator.labels_          #获取聚类标签
        centroids = estimator.cluster_centers_  #获取聚类中心
        centroids = centroids[np.lexsort(centroids.T)]  # 排个序(从小到大)
        # 表示为分子或分母1便于直观观察
        # print('\n-----ratio_cluster-----\n')
        self.cluster_res += 'ratio_cluster:\n'
        for i,c in enumerate(centroids):
            num, den = c.item().as_integer_ratio()
            if c > 1 : num /= den ; den = 1 ; num = Decimal(num).quantize(Decimal('0.00'))
            if c < 1 : den /= num ; num = 1 ; den = Decimal(den).quantize(Decimal('0.00'))
            ratio = str(num) + '/' +str(den)
            # print(ratio)
            self.cluster_res = self.cluster_res + ratio + '\n'

        plt.figure()
        plt.scatter(range(len(ratios)), ratios.squeeze(), marker='.',c=label_pred)
        plt.xlabel('gt_num')
        plt.ylabel('ratio')
        plt.title('ratio_cluster')
        for c in centroids:
            xy = np.array([int(0.5*len(boxes)),c.item()])
            plt.scatter(int(0.5*len(boxes)),c.item(), marker='*',c='red',s=100)
        plt.savefig(os.path.join(self.save_path, 'ratio_cluster.png'))
        plt.clf()

    def save_res(self):
        with open(os.path.join(self.save_path, 'cluster_results.txt'),'w') as f:
            f.write(self.cluster_res)


# 统计每个文件中物体类别数目
# CLASSES={}是类别名到数字的map便于处理
def category_statistics(labels, CLASSES=None):
    res = {}
    for classname in CLASSES:
        res[classname] = 0
    files = os.listdir(labels)
    pbar = tqdm(files)
    for filename in pbar:
        pbar.set_description("category statistics")
        with open(os.path.join(labels, filename), 'r') as f:
            objs = f.readlines()
            for obj in objs:
                classname, *_ = obj.strip().split()
                assert classname in CLASSES, 'wrong classname in '.format(filename)
                res[classname] += 1
    print(res)
    return res

if __name__ == "__main__":
    labels = r'D:\Datasets\RAChallenge\Task4\warmup\labels'
    out_dir = r'parse_result'

    # all_boxes = parse(labels, out_dir)  # 统计并保存数据集情况
    res = category_statistics(labels, 
                CLASSES = ('1', '2','3','4','5')
                            )

    # cluster_number = 5      # 该参数不一定anchor个数！，根据需要分别聚类         

    # kmeans = Kmeans(cluster_number, all_boxes, save_path=out_dir)
    
    # kmeans.anchor_clusters()
    # kmeans.area_cluster()
    # kmeans.ratio_cluster()
    # kmeans.save_res()

