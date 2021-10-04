import cv2
import glob
import os.path as osp

CLASS_NAMES = ['1', '2','3','4','5']

class RA4(object):
    def __init__(self, root_path):
        self.root_path = root_path 
        self.im_path = osp.join(root_path, 'images')
        self.anno_path = osp.join(root_path, 'labels')  
        self.im_files = glob.glob(osp.join(self.im_path,'*.tif'))
        self.anno_files = glob.glob(osp.join(self.anno_path,'*.txt'))
        self.dist_root = root_path.replace(osp.split(self.root_path)[1], osp.split(self.root_path)[1] + '_augment')
        self.dist_im_dir = osp.join(self.dist_root, 'images')
        self.dist_an_dir = osp.join(self.dist_root, 'labels')
        self.CLASSES = CLASS_NAMES

        makedir(self.dist_im_dir)
        makedir(self.dist_an_dir)

    def parse_annos(self, label):
        bboxes = []
        classnames = []
        with open(label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls, *bbox = line.strip().split()
                classnames.append(cls)
                bboxes.append([eval(x) for x in bbox])
        return classnames, np.array(bboxes)
    
    def save_labels(self, classnames, bboxes, filename):
        dist_label = osp.join(self.dist_an_dir, filename + '.txt')
        gt = ''
        for cls, bbox in zip(classnames,bboxes.tolist()):
            gt += cls + ' ' + ' '.join(str(i) for i in bbox) + '\n'
        with open(dist_label, 'w') as f:
            f.write(gt)


    def save_ims(self, im, filename):
        dist_im = osp.join(self.dist_im_dir, filename + '.tif')
        cv2.imwrite(dist_im, im)


