import numpy as np
import cv2 


def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas

def pt2poly(pt, type=None, is_Radian=None):
    if len(np.array(pt, np.int32).shape) > 1:   # 多点绘制多边形
        pts = np.array(pt, np.int32)
    elif len(pt) == 4 and type == 'xywh':
        x, y, w, h = pt   
        x1 = x - 0.5 * w 
        y1 = y - 0.5 * h 
        x2 = x + 0.5 * w 
        y2 = y + 0.5 * h
        pts = np.array([x1, y1, x2, y2], np.int32).reshape(2,2)
    elif len(pt) == 4 and type == 'xyxy':  
        pts = pt.reshape(2,2)  
    elif len(pt) == 5 and type == 'xywh':
        cx, cy, w, h, a = pt
        if is_Radian:
            a = a * 180 /3.14159
        pts = cv2.boxPoints(((cx, cy),(w,h), a))    # a为opencv定义方式，x+逆时针遇到的第一条边，顺为正 
    elif len(pt) == 5 and type == 'xyxy':
        x1, y1, x2, y2, a = pt   
        if is_Radian:
            a = a * 180 /3.14159
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = x2 - x1 
        h = y2 - y1
        pts = cv2.boxPoints(((cx, cy),(w,h), a))
    elif len(pt) == 8:
        pts = pt.reshape(4,2)
    else:
        raise RuntimeError
    return pts.astype(np.int32)



if __name__ == "__main__":
    pts = [567.44403076, 219.47401428, 355.92071533,  64.76139069, -0.89138275]
    type_pt = 'xywh'
    is_Radian = True
    canvas = InitCanvas(800, 800)
    cv2.polylines(canvas,[pt2poly(pts, type_pt, is_Radian)],True,(0,0,255))
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()