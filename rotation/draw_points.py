import numpy as np
import cv2 


def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas

def pt2poly(pt, type=None, is_Radian=None):
    if len(np.array(pt, np.int32).shape) > 1:   # 多点绘制多边形
        pts = np.array(pt, np.int32)
    elif len(pt) == 4 and 'xywh' in type :
        x, y, w, h = pt   
        x1 = x - 0.5 * w 
        y1 = y - 0.5 * h 
        x3 = x + 0.5 * w 
        y3 = y + 0.5 * h
        pts = np.array([x1, y1, x3, y1, x3, y3, x1, y3], np.int32).reshape(4,2)
    elif len(pt) == 4 and 'xyxy' in type :  
        x1, y1, x3, y3 = pt
        pts = [x1, y1, x3, y1, x3, y3, x1, y3]
        pts = np.array(pts, np.int32).reshape(4,2)  
    elif len(pt) == 5 and 'xywh' in type :
        cx, cy, w, h, a = pt
        if is_Radian:
            a = a * 180 /3.14159
        pts = cv2.boxPoints(((cx, cy),(w,h), a))    # a为opencv定义方式，x+逆时针遇到的第一条边，顺为正 
    elif len(pt) == 5 and 'xyxy' in type :
        x1, y1, x2, y2, a = pt   
        if is_Radian:
            a = a * 180 /3.14159
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = x2 - x1 
        h = y2 - y1
        pts = cv2.boxPoints(((cx, cy),(w,h), a))
    elif len(pt) == 8:
        pts = np.array(pt).reshape(4,2)
    else:
        raise RuntimeError
    return pts.astype(np.int32)



if __name__ == "__main__":
    canvas = InitCanvas(1000, 1000)

    pt1 = [285.63787842, 279.24914551,  84.1192627 ,  41.45287323, -0.8542546] 
    pt2 = [529.,  75., 574., 164.,   0.]    # cuda
    pt3 = [560.39782715,  81.24281311,  44.30744934, 135.54849243, -1.43404055]
    # canvas = cv2.polylines(canvas,[pt2poly(pt1, 'xyxy', is_Radian= True)], True,(0,0,255))
    canvas = cv2.polylines(canvas,[pt2poly(pt2, 'xyxy', is_Radian= True)], True,(0,0,255))    
    canvas = cv2.polylines(canvas,[pt2poly(pt3, 'xywh', is_Radian= True)], True,(0,0,255))

    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    