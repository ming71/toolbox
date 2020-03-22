import os 
import cv2
import numpy as np
import shutil
import math
from tqdm import tqdm
from decimal import Decimal

# 注意文件结构！dets和gt的目录是map测试文件所在的目录下
# 由于文本的不重叠性，直接在所有result加上conf = 0.99

def rbox_2_quad(rbox):
    quads = np.zeros((8), dtype=np.float32)
    x = rbox[0]
    y = rbox[1]
    w = rbox[2]
    h = rbox[3]
    theta = rbox[4]
    quads= cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))
    return quads


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted


def class_mapping(cls_id, level):
    if level == 1:
        classes = ('__background__', 'ship') 
        return 'ship'
    if level == 2:
        classes = ('__background__', 'ship', 'air.', 'war.','mer.') 
        if cls_id in ['100000005','100000006','100000012','100000013','100000016','10000032']:
            cls_id =  '100000002'
        if cls_id in ['100000007','100000008','100000009','100000010','100000011','10000015',
                        '10000017','10000019','10000028','10000029']:
            cls_id =  '100000003'
        if cls_id in ['100000018','100000020','100000024','100000025','100000026','10000030']:
            cls_id = '100000004'
        class_ID = ['bg', '100000001', '100000002', '100000003', '100000004']
        return classes[class_ID.index(cls_id)]
    if self.level == 3:
        classes = ('__background__', 'ship' , 'air.', 'war.','mer.', 'Nim.', 
                        'Ent.' , 'Arl.' , 'Whi.' , 'Per.' , 'San.' , 'Tic.' , 'Aus.' , 
                        'Tar.' , 'Con.' , 'Com.A' , 'Car.A' , 'Con.A' , 'Med.' , 'Car.B' ) 
        if cls_id in ['1000000012', '1000000013', '1000000032',]:
            cls_id = '100000002'
        if cls_id in ['100000017', '100000028']:
            cls_id = '100000003'
        if cls_id in ['100000024', '100000026']:
            cls_id = '100000004'
        class_ID = ['bg', '100000001', '100000002', '100000003', '100000004', '100000005', 
                    '100000006', '100000007', '100000008', '100000009', '1000000010',
                    '1000000011', '100000015','1000000016', '100000018', '100000019', 
                    '100000020', '100000025' , '100000029', '100000030'] 
        return classes[class_ID.index(cls_id)]


def convert_gt(gt_path, dst_path, level=1):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    gts = os.listdir(gt_path)
    files = [os.path.join(gt_path, x) for x in gts]
    dst_gt = [os.path.join(dst_path, x.replace('.xml','.txt')) for x in gts]
    print('gt generating...')
    for i, filename in enumerate(tqdm(files)):
        with open(filename,'r',encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('<HRSC_Object>')
            info = objects.pop(0)
            nt = 0
            for obj in objects:
                assert len(obj) != 0, 'No onject found in %s'%filename
                cls_id = obj[obj.find('<Class_ID>')+10 : obj.find('</Class_ID>')]
                diffculty = obj[obj.find('<difficult>')+11 : obj.find('</difficult>')]
                if cls_id in ['100000027', '100000022'] :
                    continue
                nt += 1
                cx = (eval(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')]))
                cy = (eval(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')]))
                w  = (eval(obj[obj.find('<mbox_w>')+8 : obj.find('</mbox_w>')]))
                h  = (eval(obj[obj.find('<mbox_h>')+8 : obj.find('</mbox_h>')]))
                a  = eval(obj[obj.find('<mbox_ang>')+10 : obj.find('</mbox_ang>')])/math.pi*180
                pt = sort_corners(rbox_2_quad([cx,cy,w,h,a]))[0]    # xyxyxyxy
                name = class_mapping(cls_id,level)
                with open(dst_gt[i],'a') as fd:
                    if diffculty == '0':
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                            name,pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7]
                        ))
                    elif diffculty == '1':
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} difficult\n'.format(
                            name,pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7]
                        ))
                    else:
                        raise RuntimeError('???? difficult wronging!!')                    
            if nt == 0:
                os.remove(filename)
                os.remove(filename.replace('Annotations','AllImages').replace('xml','bmp'))



if __name__ == "__main__":
    level = 1
    gt_path = '/py/BoxesCascade/HRSC2016/Train/Annotations' # 给定的gt文件夹
    dst_path = '/py/BoxesCascade/test/ground-truth'

    convert_gt(gt_path, dst_path,level)

    

