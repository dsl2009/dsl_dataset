import glob
from matplotlib import pyplot as plt
import os
import json
from dsl_data.utils import resize_image_fixed_size
import cv2
from dsl_data.visual import display_instances
from skimage import io
import numpy as np
class MianHua(object):
    def __init__(self, dirs, image_size):
        self.dirs = dirs
        self.files = glob.glob(os.path.join(self.dirs,'*.json'))
        self.image_size = image_size
        self.index = range(len(self.files))
        self.class_to_idx = {'open':0, 'un_open':1,'unopen':1}
    def pull_item(self,idx):
        file = self.files[self.index[idx]]
        file = file.replace('\\','/')
        image = file.replace('.json','.jpg')

        json_data = json.loads(open(file).read())
        image = io.imread(image)
        image, window, scale, padding, crop = resize_image_fixed_size(image, self.image_size)
        label_ix = []
        box = []
        for s in json_data['shapes']:
            label_ix.append(self.class_to_idx[s['label']])
            box.append([s['points'][0][0],s['points'][0][1],s['points'][2][0],s['points'][2][1]])
        box = np.asarray(box)
        box = box * scale
        box[:, 0] = box[:, 0] + padding[1][0]
        box[:, 1] = box[:, 1] + padding[0][0]
        box[:, 2] = box[:, 2] + padding[1][1]
        box[:, 3] = box[:, 3] + padding[0][1]

        box =  box / np.asarray([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])
        return image, box, label_ix
    def len(self):
        return len(self.index)





