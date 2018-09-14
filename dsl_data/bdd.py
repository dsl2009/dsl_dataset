import json
import os
import glob
from skimage import io
import numpy as np
from dsl_data import visual, utils
from matplotlib import pyplot as plt
classes = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck']

class BDD(object):
    def __init__(self, js_file, image_dr, image_size):
        self.js_file = js_file
        self.image_dr = image_dr
        self.image_size = image_size
        self.data = json.loads(open(self.js_file).read())
        self.class_mapix = dict(zip(classes,range(len(classes))))
        print(self.class_mapix)
    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        item_data = self.data[idx]
        image_name = item_data['name']
        labels = item_data['labels']
        label_ix = []
        box = []
        ig_data = io.imread(os.path.join(self.image_dr, image_name))
        for ll in labels:

            category_name = ll['category']

            if self.class_mapix.get(category_name,None):
                box2d = ll['box2d']

                label_ix.append(self.class_mapix[category_name])
                box.append([box2d['x1'],box2d['y1'],box2d['x2'],box2d['y2']])

        box = np.asarray(box)
        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig_data, self.image_size)

        box = box * scale
        box[:, 0] = box[:, 0] + padding[1][0]
        box[:, 1] = box[:, 1] + padding[0][0]
        box[:, 2] = box[:, 2] + padding[1][1]
        box[:, 3] = box[:, 3] + padding[0][1]
        box = box/np.asarray([self.image_size[0], self.image_size[1],self.image_size[0], self.image_size[1]])
        return ig, box, label_ix






def tt():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train'
    js_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json'
    data_set = BDD(js_file=js_file, image_dr = image_dr, image_size = [768, 1280])
    for k in range(10):
        data_set.pull_item(k)
