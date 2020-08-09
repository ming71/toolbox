import cv2
import torch
import numpy as np

def xy2wh(boxes):
    """
    :param boxes: (xmin, ymin, xmax, ymax) (n, 4)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
    """
    if torch.is_tensor(boxes):
        out_boxes = boxes.clone()
    else:
        out_boxes = boxes.copy()
    out_boxes[:, 2] = boxes[:, 2] - boxes[:, 0] + 1.0
    out_boxes[:, 3] = boxes[:, 3] - boxes[:, 1] + 1.0
    out_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5

    return out_boxes


def wh2xy(boxes):
    """
    :param boxes: (x_ctr, y_ctr, w, h) (n, 4)
    :return: out_boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    if torch.is_tensor(boxes):
        out_boxes = boxes.clone()
    else:
        out_boxes = boxes.copy()
    out_boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] - 1.0) * 0.5
    out_boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] - 1.0) * 0.5
    out_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] * 0.5
    out_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5

    return out_boxes




def rbox2poly(boxes, is_Radian=True):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, theta
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    if torch.is_tensor(boxes):
        out_boxes = boxes.clone()
    else:
        out_boxes = boxes.copy()
    
    if not is_Radian:
        out_boxes[:, 4] = out_boxes[:, 4] * 3.1415926 / 180

    cs = np.cos(out_boxes[:, 4])
    ss = np.sin(out_boxes[:, 4])
    w = out_boxes[:, 2] - 1
    h = out_boxes[:, 3] - 1

    ## change the order to be the initial definition
    x_ctr = out_boxes[:, 0]
    y_ctr = out_boxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)
    
    if torch.is_tensor(boxes):
        polys = torch.cat((x1.unsqueeze(1),
                        y1.unsqueeze(1),
                        x2.unsqueeze(1),
                        y2.unsqueeze(1),
                        x3.unsqueeze(1),
                        y3.unsqueeze(1),
                        x4.unsqueeze(1),
                        y4.unsqueeze(1)), 1)
    else:
        x1 = x1[:, np.newaxis]
        y1 = y1[:, np.newaxis]
        x2 = x2[:, np.newaxis]
        y2 = y2[:, np.newaxis]
        x3 = x3[:, np.newaxis]
        y3 = y3[:, np.newaxis]
        x4 = x4[:, np.newaxis]
        y4 = y4[:, np.newaxis]
        polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)

    return polys

# only support numpy outputs now
def poly2rbox(bbox, toRadian=True):
    """
    :param bbox:  [x1, y1, x2, y2, x3, y3, x4, y4]    [num_boxes, 8]
    :return: [cx, cy, w, h, theta]   [num_rot_recs, 5]
    """
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])

    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)

    xmin = np.min(normalized[:, 0, :], axis=1)
    xmax = np.max(normalized[:, 0, :], axis=1)
    ymin = np.min(normalized[:, 1, :], axis=1)
    ymax = np.max(normalized[:, 1, :], axis=1)

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes

if __name__ == "__main__":
    # boxes = np.array([[100.,100.,50.,60.,20.]])
    boxes = np.array([[133.1121,  80.6586, 112.9329, 136.1004,  66.8879, 119.3414,  87.0671,63.8996]])
    boxes = torch.from_numpy(boxes)
    _boxes = poly2rbox(boxes, False)
    print(boxes)
    print(type(_boxes))
    print(0.3490665 /3.14*180)