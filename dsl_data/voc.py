import os.path as osp
import sys
from dsl_data import utils
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import numpy as np
from skimage import io
from dsl_data import visual

classes = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
class VOCDetection(object):
    def __init__(self, root, image_size,is_crop=False, image_sets=[ ('2007', 'trainval'),('2012', 'trainval')]):
        self.root = root
        self.image_set = image_sets
        self.image_size = image_size
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.class_mapix = dict(zip(classes, range(len(classes))))
        self.ids = list()
        self.is_crop = is_crop
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
    def len(self):
        return len(self.ids)
    def pull_item(self, index):
        img_id = self.ids[index]
        boxes = []
        labels = []
        target = ET.parse(self._annopath % img_id).getroot()
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if False and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            if self.class_mapix.get(name, None) is not None:
                label_idx = self.class_mapix[name]
                labels.append(label_idx)
                #labels.append(0)
                boxes.append(bndbox)
        if len(boxes) == 0:
            return None, None, None
        img = io.imread(self._imgpath % img_id)
        box = np.asarray(boxes)
        if not self.is_crop:
            ig, window, scale, padding, crop = utils.resize_image_fixed_size(img, self.image_size)
            if len(labels)==0:
                return ig, box, labels
            box = box * scale
            box[:, 0] = box[:, 0] + padding[1][0]
            box[:, 1] = box[:, 1] + padding[0][0]
            box[:, 2] = box[:, 2] + padding[1][1]
            box[:, 3] = box[:, 3] + padding[0][1]

        else:
            ig, box, labels = utils.crop_image_with_box(img, self.image_size, box, labels)
        box = box / np.asarray([self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]])
        return ig, box, labels



def tt():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit'
    data_set = VOCDetection(root=image_dr,is_crop=False,  image_size = [512, 512])
    ttlb = []
    image_size = [512, 512]

    for x in range(100):
        result = data_set.pull_item(x)
        if result:
            ig, box, labels = data_set.pull_item(x)
            if len(labels) >0 :
                box = box * np.asarray([image_size[1], image_size[0], image_size[1], image_size[0]])
                print(box)
                visual.display_instances(ig,box)
