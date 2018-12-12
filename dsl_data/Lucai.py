#coding=utf-8
import json
import os
import glob
from skimage import io
import numpy as np
from dsl_data import visual, utils
from matplotlib import pyplot as plt
import cv2
from dsl_data import aug_utils
from dsl_data import coco_handler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
classes = ['不导电', '擦花', '角位漏底', '桔皮', '漏底', '喷流', '漆泡', '起坑', '杂色', '脏点']


class Lucai(object):
    def __init__(self, image_dr, image_size, is_crop=True):
        self.image_dr = image_dr
        self.image_size = image_size
        l1 = glob.glob(os.path.join(image_dr,'*','*.jpg'))
        l2 = glob.glob(os.path.join(image_dr,'*','*','*.jpg'))
        l1.extend(l2)
        self.data = l1
        self.class_mapix = dict(zip(classes,range(len(classes))))
        self.is_crop = is_crop
        print(self.class_mapix)
    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        image_path = self.data[idx]
        js_path = self.data[idx].replace('.jpg','.json')
        js_data = json.loads(open(js_path).read())

        label_ix = []
        box = []
        ig_data = io.imread(image_path)
        for b in js_data:
            category_name = b['label']
            if self.class_mapix.get(category_name, None) is not None:
                bound = np.asarray(b['points'])
                minx, miny, maxx, maxy = min(bound[:, 0]), min(bound[:, 1]), max(bound[:, 0]), max(bound[:, 1]),
                box.append([minx, miny, maxx, maxy])
                label_ix.append(self.class_mapix[category_name])
        box = np.asarray(box)
        if box.shape[0] == 0:
            return None,None,None
        if self.is_crop:
            ig, box, label_ix = utils.crop_image_with_box(ig_data, self.image_size, box, label_ix)

        else:
            ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig_data, self.image_size)
            box = box * scale
            box[:, 0] = box[:, 0] + padding[1][0]
            box[:, 1] = box[:, 1] + padding[0][0]
            box[:, 2] = box[:, 2] + padding[1][1]
            box[:, 3] = box[:, 3] + padding[0][1]

        box = box/np.asarray([self.image_size[1], self.image_size[0],self.image_size[1], self.image_size[0]])
        return ig, box, label_ix
def tt():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/round2'
    image_size = [896, 1024]
    data_set = Lucai(image_dr,  image_size=image_size,is_crop=False)
    ttlb = []

    index = range(data_set.len())
    index = np.asarray(index)
    np.random.shuffle(index)
    for x in index:
        result = data_set.pull_item(x)
        if result:
            ig, box, labels = data_set.pull_item(x)
            if labels is not None and len(labels) > 0:
                ig, box = aug_utils.fliplr_up_down(ig, box)
                box = box * np.asarray([image_size[1], image_size[0],image_size[1], image_size[0]])
                visual.display_instances(ig, box)
