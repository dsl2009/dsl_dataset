import glob
import os
import cv2
import numpy as np
import json
from dsl_data import utils
from dsl_data import visual
from matplotlib import pyplot as plt
import random
from imgaug import augmenters as iaa
from torchvision.transforms import transforms
from dsl_data import aug_utils
import imgaug as ia

class Uav(object):
    def __init__(self,image_size):
        self.root_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/xiaohuan/'
        self.image_size = image_size
        self.images = glob.glob(os.path.join(self.root_dr, '*', '*.jpg'))
        self.img_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.ContrastNormalization((0.7, 1.5)),
            iaa.Multiply((0.7, 1.5)),
            iaa.Affine(
                scale=(0.8, 1.6)
            )
        ])
        self.labels_index = {'uav':0,'qr':1}

    def len(self):
        return len(self.images)

    def pull_item(self, index):
        image_path = self.images[index]
        json_path = image_path.replace('.jpg', '.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = []
        box = []
        for x in bundry['shapes']:
            pt = x['points']
            labels.append(self.labels_index[x['label']])
            box.append([pt[0][0],pt[0][1],pt[2][0],pt[2][1]])
        box = np.asarray(box)
        if False:
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
        bb = []
        for ix, x in enumerate(box):
            bb.append(ia.BoundingBox(x[0],x[1],x[2],x[3],labels[ix]))
        bbs = ia.BoundingBoxesOnImage(bb, shape=self.image_size)
        seq_det = self.img_aug.to_deterministic()
        image_aug = seq_det.augment_images([ig])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        box = []
        labels = []
        for i in range(len(bbs.bounding_boxes)):
            after = bbs_aug.bounding_boxes[i]
            box.append([after.x1, after.y1, after.x2, after.y2])
            labels.append(after.label)
        box = np.asarray(box)
        box = box / np.asarray([self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]])
        box = np.clip(box, 0,1)
        return image_aug, box, labels


if __name__ == '__main__':
    image_size = [512,512]
    data_set = Uav(image_size=image_size)
    for x in range(100):
        ig, box, labels = data_set.pull_item(x)
        if len(labels) > 0:
            box = box * np.asarray([image_size[1], image_size[0], image_size[1], image_size[0]])
            print(box)
            visual.display_instances(ig, box)






