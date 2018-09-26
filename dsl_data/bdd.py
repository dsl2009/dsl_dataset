import json
import os
import glob
from skimage import io
import numpy as np
from dsl_data import visual, utils
from matplotlib import pyplot as plt
import cv2
from dsl_data import coco_handler
classes = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck']


class BDD(object):
    def __init__(self, js_file, image_dr, image_size, is_crop=False):
        self.js_file = js_file
        self.image_dr = image_dr
        self.image_size = image_size
        self.data = json.loads(open(self.js_file).read())
        self.class_mapix = dict(zip(classes,range(len(classes))))
        self.is_crop = is_crop
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
            if self.class_mapix.get(category_name,None) is not None:
                box2d = ll['box2d']
                label_ix.append(self.class_mapix[category_name])
                box.append([box2d['x1'],box2d['y1'],box2d['x2'],box2d['y2']])

        box = np.asarray(box)
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


class BDD_AREA(object):
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
        direct = np.zeros(shape=ig_data.shape,dtype= np.uint8)
        alter = np.zeros(shape=ig_data.shape, dtype=np.uint8)

        for ll in labels:
            category_name = ll['category']
            if category_name == 'drivable area':
                if ll['attributes']['areaType'] == 'direct':
                    pts = ll['poly2d']
                    for x in pts:
                        cv2.fillPoly(direct, [np.asarray(x['vertices'], np.int)] ,(255, 255, 255))
                elif ll['attributes']['areaType'] == 'alternative':
                    pts = ll['poly2d']
                    for x in pts:
                        cv2.fillPoly(alter, [np.asarray(x['vertices'], np.int)], (255, 255, 255))

        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig_data, self.image_size)
        direct, window, scale, padding, crop = utils.resize_image_fixed_size(direct, self.image_size)
        alter, window, scale, padding, crop = utils.resize_image_fixed_size(alter, self.image_size)

        labels = np.zeros(shape=(ig.shape[0],ig.shape[1],2),dtype=np.uint8)
        labels[:, :, 0] = direct[:, :, 0]
        labels[:, :, 1] = alter[:, :, 0]
        return ig, labels

class BDD_AREA_MASK(object):
    def __init__(self, js_file, image_dr, image_size,mask_shape = 28):
        self.js_file = js_file
        self.image_dr = image_dr
        self.image_size = image_size
        self.data = json.loads(open(self.js_file).read())
        self.class_mapix = dict(zip(classes,range(len(classes))))
        self.mask_shape = mask_shape
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

        instance_masks = []
        cls_ids = []
        for ll in labels:
            category_name = ll['category']
            if category_name == 'drivable area':
                if ll['attributes']['areaType'] == 'direct':
                    pts = ll['poly2d']
                    for x in pts:
                        direct = np.zeros(shape=ig_data.shape, dtype=np.uint8)
                        cv2.fillPoly(direct, [np.asarray(x['vertices'], np.int)] ,(255, 255, 255))
                        cls_ids.append(0)
                        instance_masks.append(direct[:,:,0])
                elif ll['attributes']['areaType'] == 'alternative':
                    pts = ll['poly2d']
                    for x in pts:
                        alter = np.zeros(shape=ig_data.shape, dtype=np.uint8)
                        cv2.fillPoly(alter, [np.asarray(x['vertices'], np.int)], (255, 255, 255))
                        cls_ids.append(1)
                        instance_masks.append(alter[:, :, 0])
        mask = np.asarray(instance_masks)
        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig_data, self.image_size)
        if len(labels) == 0:
            return
        mask = np.transpose(mask, axes=[1, 2, 0])
        mask = utils.resize_mask(mask, scale, padding, crop)
        image, mask = coco_handler.aug(ig, mask)

        boxes = utils.extract_bboxes(mask)
        mask = utils.minimize_mask(boxes, mask, mini_shape=(self.mask_shape, self.mask_shape))
        #boxes = boxes /np.asarray([self.image_size[1], self.image_size[0],self.image_size[1], self.image_size[0]])
        return ig, cls_ids, boxes, mask







def tt():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train'
    js_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json'
    data_set = BDD(js_file=js_file, image_dr = image_dr, image_size = [512, 512])
    ttlb = []

    for x in range(100):
        data_set.pull_item(x)

def get_class_num():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train'
    js_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json'
    data_set = BDD_AREA_MASK(js_file=js_file, image_dr = image_dr, image_size = [768, 1280])
    ttlb = []

    for x in range(100):
        ig, cls_ids, boxes, mask = data_set.pull_item(x)
        print(cls_ids)
        visual.display_instances(ig, boxes)

