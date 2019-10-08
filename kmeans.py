'''
2019.10.8  ming71
功能:  对box进行kmeans聚类
注意:  
    - 停止条件是最小值索引不变而不是最小值不变，会造成早停，可以改
    - 暂时仅支持voc标注
    - 如需改动再重写get_all_boxes函数即可
'''
import numpy as np
import glob
import os
from decimal import Decimal

class Kmeans:

    def __init__(self, cluster_number, all_boxes, save_path):
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
    
    #注意：这里代码选择的停止聚类的条件是最小值的索引不变，而不是种子点的数值不变。这样的误差会大一点。

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]                 # box个数
        distances = np.empty((box_number, k))       # 初始化[box_number , k]二维数组，存放自定义iou距离（obj*anchor）
        last_nearest = np.zeros((box_number,))       # [box_number , ]的标量
        np.random.seed()                           
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # 种子点随机初始化

        # 种子点一旦重复会有计算错误,避免!   
        while True :
            uniques_clusters = np.unique(clusters,axis=0)
            if  len(uniques_clusters)==len(clusters) :
                break
            clusters = boxes[np.random.choice(box_number, k, replace=False)]
        
        # k-means
        while True:
            # 每轮循环，计算种子点外所有点各自到k个种子点的自定义距离，并且按照距离各个点找离自己最近的种子点进行归类；计算新的各类中心；然后下一轮循环
            distances = 1 - self.iou(boxes, clusters)   # iou越大,距离越小

            current_nearest = np.argmin(distances, axis=1)  # 展开为box_number长度向量,代表每个box当前属于哪个种子点类别(0,k-1) 
            if (last_nearest == current_nearest).all():     # 每个box的当前类别所属和上一次相同,不再移动聚类
                break                                       

            #计算新的k个种子点坐标
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0) # 只对还需要聚类的种子点进行位移
            last_nearest = current_nearest

        return clusters

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


    def clusters(self):
        all_boxes = np.array(self.all_boxes)                                #返回全部gt的宽高二维数组
        result = self.kmeans(all_boxes, k=self.cluster_number)      #传入两个聚类参数：所有gt宽高的二维数组和种子点数，并返回聚类结果k*2
        result = result[np.lexsort(result.T[0, None])]              #将得到的三个anchor按照宽进行从小到大，重新排序
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))



# 返回所有label的box,形式为[[w1,h1],[w2,h2],...]
def get_all_boxes(path):
    mode = 'voc'
    boxes = []

    labels = sorted(glob.glob(os.path.join(path, '*.*')))
    for label in labels:
        with open(label,'r') as f:
            contents = f.read()
            objects = contents.split('<object>')
            objects.pop(0)
            assert len(objects) > 0, 'No object found in ' + xml_path

            for object in objects:
                xmin = int(object[object.find('<xmin>')+6 : object.find('</xmin>')])
                xmax = int(object[object.find('<xmax>')+6 : object.find('</xmax>')])
                ymin = int(object[object.find('<ymin>')+6 : object.find('</ymin>')])
                ymax = int(object[object.find('<ymax>')+6 : object.find('</ymax>')])
                box_w = xmax - xmin 
                box_h = ymax - ymin
                boxes.append((box_w,box_h))
    return boxes


if __name__ == "__main__":
    cluster_number = 9              # 种子点个数,即anchor数目
    label_path = r'/py/datasets/ship/tiny_ships/yolo_ship/train_labels'
    save_path = r'/py/yolov3/cfg/anchor-cluster.txt'

    all_boxes = get_all_boxes(label_path)   
    kmeans = Kmeans(cluster_number, all_boxes,save_path)
    kmeans.clusters()
