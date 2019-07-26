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
import imgaug as ia
from dsl_data import aug_utils
from skimage import io

class Face(object):
    def __init__(self, image_size):
        self.img_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.ContrastNormalization((0.7, 1.5)),
            iaa.Multiply((0.7, 1.5)),
            iaa.Affine(
                scale=(0.8, 1.6)
            )
        ])
        self.image_size = image_size
        self.image_path = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/WIDER_train/images'
        self.images, self.boxes = self.handler()

    def len(self):
        return len(self.images)

    def handler(self):
        label_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/wider_face_split/wider_face_train_bbx_gt.txt'
        images = []
        total_boxes =[]
        boxes = None
        with open(label_file,'r') as f:
            contents = f.readlines()
            for x in contents:
                x = x.replace('\n','')
                if '.jpg' in x:
                    images.append(x)
                    if boxes is not None and len(boxes)>0:
                        total_boxes.append(boxes)
                    boxes = []
                elif len(x)>10 and len(x.split(' '))<200:
                    dts = x.split(' ')
                    x1, y1, w, h = int(dts[0]),int(dts[1]),int(dts[2]),int(dts[3])
                    x2, y2 = x1+w,y1+h
                    boxes.append([x1, y1, x2, y2])
        return images, total_boxes

    def pull_item(self, index):
        img_id = os.path.join(self.image_path, self.images[index])
        boxes = self.boxes[index]
        labels = np.ones(len(boxes))*0

        if len(boxes) == 0:
            return None, None, None
        img = io.imread(img_id)
        box = np.asarray(boxes)
        if True:
            ig, window, scale, padding, crop = utils.resize_image_fixed_size(img, self.image_size)
            if len(labels)==0:
                return ig, box, labels
            box = box * scale
            box[:, 0] = box[:, 0] + padding[1][0]
            box[:, 1] = box[:, 1] + padding[0][0]
            box[:, 2] = box[:, 2] + padding[1][1]
            box[:, 3] = box[:, 3] + padding[0][1]

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
    image_size = [512, 512]
    face = Face(image_size=image_size)
    for x in range(100):
        ig, box, labels = face.pull_item(x)
        if len(labels) > 0:
            box = box * np.asarray([image_size[1], image_size[0], image_size[1], image_size[0]])
            print(box)
            visual.display_instances(ig, box)
