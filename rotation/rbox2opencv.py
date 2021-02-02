import numpy as np
import math

def dist(p1, p2):
    return  np.linalg.norm(p1 - p2, axis=1)


def poly2rbox(boxes):
    # input: ordered points (bs, 8) 
    nB = len(boxes)
    points = boxes.reshape(-1, 4, 2)
    cxs = points[:,:,0].sum(1)[:,np.newaxis] / 4
    cys = points[:,:,1].sum(1)[:,np.newaxis] / 4
    _ws = dist(points[:,0], points[:,1])[:,np.newaxis]
    _hs = dist(points[:,1], points[:,2])[:,np.newaxis]
    # adjust theta
    _thetas = np.arctan2(-(points[:,1,0]-points[:,0,0]),points[:,1,1]-points[:,0,1])[:, np.newaxis]
    odd = (np.mod((_thetas // (-np.pi*0.5)), 2) == 0)
    ws = np.where(odd, _hs, _ws)
    hs = np.where(odd, _ws, _hs)
    thetas = np.mod(_thetas, -np.pi*0.5)
    rboxes = np.concatenate([cxs, cys, ws, hs, thetas], 1)

    return rboxes



if __name__ == "__main__":
    box = poly2rbox(np.array([[278, 418,308, 331 ,761, 581, 691, 668],
    [758, 418,348, 331 ,241, 581, 591, 668],
    [624, 112,490,  93,496,  50,630,  68]
    ]))
    print(box)