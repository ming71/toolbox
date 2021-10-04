import os
import shutil
from tqdm import tqdm
from shapely.geometry import Polygon


def makedir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    
# 统计每个文件中物体类别数目
# CLASSES是类别名
def category_statistics(labels, CLASSES=None):
    res_cnt = {}
    box_areas = []
    for classname in CLASSES:
        res_cnt[classname] = 0
    pbar = tqdm(labels)
    for label in pbar:
        pbar.set_description("category statistics")
        with open(label, 'r') as f:
            objs = f.readlines()
            for obj in objs:
                classname, *bbox = obj.strip().split()
                box_areas.append(bbox_area(bbox))
                assert classname in CLASSES, 'wrong classname in '.format(filename)
                res_cnt[classname] += 1
    return res_cnt, box_areas


def augment_ratio(cnt):
    objects = [x for x in  cnt.values()]
    rates = [int(max(objects) / x) - 1 for x in objects]
    rates[0] = int(rates[0]/10)  # [117, 857, 756, 904, 574]
    scheduler = cnt.copy()
    for idx, classname in enumerate(scheduler.keys()):
        scheduler[classname] = rates[idx]
    print(scheduler)
    return scheduler


def bbox_area(bbox):
    bbox = np.asarray(bbox)
    bbox = np.array(bbox).reshape(4, 2)
    poly = Polygon(bbox).convex_hull
    return poly.area


## bbox trans 
def quad_2_rbox(quads, mode='xyxya'):
    # http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg
    if len(quads.shape) == 1:
        quads = quads[np.newaxis, :]
    rboxes = np.zeros((quads.shape[0], 5), dtype=np.float32)
    for i, quad in enumerate(quads):
        rbox = cv2.minAreaRect(quad.reshape([4, 2]))    
        x, y, w, h, t = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]
        if np.abs(t) < 45.0:
            rboxes[i, :] = np.array([x, y, w, h, t])
        elif np.abs(t) > 45.0:
            rboxes[i, :] = np.array([x, y, h, w, 90.0 + t])
        else:   
            if w > h:
                rboxes[i, :] = np.array([x, y, w, h, -45.0])
            else:
                rboxes[i, :] = np.array([x, y, h, w, 45])
    # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    if mode == 'xyxya':
        rboxes[:, 0:2] = rboxes[:, 0:2] - rboxes[:, 2:4] * 0.5
        rboxes[:, 2:4] = rboxes[:, 0:2] + rboxes[:, 2:4]
    rboxes[:, 0:4] = rboxes[:, 0:4].astype(np.int32)
    return rboxes

def rbox_2_quad(rboxes, mode='xyxya'):
    if len(rboxes.shape) == 1:
        rboxes = rboxes[np.newaxis, :]
    if rboxes.shape[0] == 0:
        return rboxes
    quads = np.zeros((rboxes.shape[0], 8), dtype=np.float32)
    for i, rbox in enumerate(rboxes):
        if len(rbox!=0):
            if mode == 'xyxya':
                w = rbox[2] - rbox[0]
                h = rbox[3] - rbox[1]
                x = rbox[0] + 0.5 * w
                y = rbox[1] + 0.5 * h
                theta = rbox[4]
            elif mode == 'xywha':
                x = rbox[0]
                y = rbox[1]
                w = rbox[2]
                h = rbox[3]
                theta = rbox[4]
            quads[i, :] = cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))

    return quads

