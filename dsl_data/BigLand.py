import glob
import json
import os
from skimage import io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa
import imgaug as ia
from dsl_data import visual
from dsl_data import utils
import random
labels = ['land', 'canal', 'pond', 'tree', 'other', 'building']
color = [(255,255,0),(0,191,255),(0,0,255), (34,139,34),(245,245,245),(255,0,0)]
class BigLand(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*.json')
        self.class_mapix = dict(zip(labels,range(len(labels))))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),
        ])
    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        json_pth = self.data[idx]
        image_pth = json_pth.replace('.json','.png')
        ig_data = io.imread(image_pth)[:,:,0:3]
        instance_masks = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
        js_data = json.loads(open(json_pth).read())
        for b in js_data['boundary']:
            label = b['correctionType']
            if label == 'land':
                points = b['points']
                p = []
                for pp in points:
                    p.append([pp['pix_x'],pp['pix_y']])
                label_id = self.class_mapix[label]
                direct = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
                cv2.fillPoly(direct, np.asarray([p], np.int),(255,255,255))
                #cv2.polylines(direct,np.asarray([p], np.int),True, (255,255,255), thickness=2)
                instance_masks += direct

        if random.randint(0,1) ==1:
            ag = self.aug.to_deterministic()
            ig_data = ag.augment_image(ig_data)
            instance_masks = ag.augment_image(instance_masks)

        return ig_data, instance_masks[:,:,0:1]

class BigLandArea(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*.json')
        self.class_mapix = dict(zip(labels,range(len(labels))))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),
        ])
    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        json_pth = self.data[idx]
        image_pth = json_pth.replace('.json','.png')
        ig_data = io.imread(image_pth)[:,:,0:3]
        instance_masks = []
        js_data = json.loads(open(json_pth).read())
        for b in js_data['boundary']:
            label = b['correctionType']
            if label == 'land':
                points = b['points']
                p = []
                for pp in points:
                    p.append([pp['pix_x'],pp['pix_y']])
                label_id = self.class_mapix[label]
                direct = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
                cv2.fillPoly(direct, np.asarray([p], np.int),(255,255,255))
                #cv2.polylines(direct,np.asarray([p], np.int),True, (255,255,255), thickness=2)
                instance_masks.append(direct[:,:,0:1])

        instance_masks = np.concatenate(instance_masks, axis=2)

        if random.randint(0,1) ==1:
            ag = self.aug.to_deterministic()
            ig_data = ag.augment_image(ig_data)
            instance_masks = ag.augment_image(instance_masks)

        seg_mask = np.sum(instance_masks, axis=2)
        seg_mask[np.where(seg_mask>255)] = 255


        return ig_data, instance_masks, seg_mask, instance_masks.shape[2]



class BigLandMask(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*.json')
        self.class_mapix = dict(zip(labels,range(len(labels))))

    def len(self):
        return len(self.data)
    def is_right(self,pts):
        d = 0
        for x in range(len(pts)-1):
            y1, x1 = int(pts[x]['pix_y']),int(pts[x]['pix_x'])
            y2, x2 = int(pts[x+1]['pix_y']), int(pts[x+1]['pix_x'])

            d += -0.5 * (y2 + y1) * (x2 - x1);
        return d



    def pull_item(self,idx):
        json_pth = self.data[idx]
        image_pth = json_pth.replace('.json','.png')
        ig_data = io.imread(image_pth)[:,:,0:3]
        shape = ig_data.shape[0:2]

        instance_masks = []
        js_data = json.loads(open(json_pth).read())
        ids = []
        mask = np.zeros(shape=(shape[0], shape[1], 5), dtype=np.float32)

        for b in js_data['boundary']:
            label = b['correctionType']
            if label == 'land':

                boun = b['points'][0:-1]
                #print(self.is_right(b['points']))
                p = []
                for ix, p in enumerate(boun):
                    tm_msk = np.zeros(shape=(shape[0], shape[1], 3), dtype=np.uint8)
                    #mask[int(p['pix_x']), int(p['pix_y']), 0] = 1.0

                    cv2.circle(tm_msk, center=(int(p['pix_x']), int(p['pix_y'])),radius=3, color=(255,255,255), thickness=-1)
                    mask[:,:,0] = mask[:,:,0]+tm_msk[:,:,0]/255.0


                    left = boun[ix - 1]
                    if ix == len(boun) - 1:
                        right = boun[0]
                    else:
                        right = boun[ix + 1]

                    mask[int(p['pix_x']), int(p['pix_y']), 1] = 2*float(left['pix_x']) / self.image_size[0]-1.0
                    mask[int(p['pix_x']), int(p['pix_y']), 2] = 2*float(left['pix_y']) / self.image_size[1]-1.0

                    mask[int(p['pix_x']), int(p['pix_y']), 3] = 2*float(right['pix_x']) / self.image_size[0]-1.0
                    mask[int(p['pix_x']), int(p['pix_y']), 4] = 2*float(right['pix_y']) / self.image_size[1]-1.0
        mask[np.where(mask>1)] = 1.0

        return ig_data, mask

class BigLandMaskCoc(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*.json')
        self.class_mapix = dict(zip(labels,range(len(labels))))

    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        json_pth = self.data[idx]
        image_pth = json_pth.replace('.json','.png')
        ig_data = io.imread(image_pth)[:,:,0:3]
        shape = ig_data.shape[0:2]

        instance_masks = []
        js_data = json.loads(open(json_pth).read())
        ids = []
        mask = np.zeros(shape=(shape[0], shape[1], 3), dtype=np.uint8)

        for b in js_data['boundary']:
            label = b['correctionType']
            if label == 'land':

                boun = b['points'][0:-1]
                p = []
                for ix, p in enumerate(boun):
                    p0 = [int(p['pix_x']), int(p['pix_y'])]

                    left = boun[ix - 1]
                    if ix == len(boun) - 1:
                        right = boun[0]
                    else:
                        right = boun[ix + 1]

                    end_l = [int(left['pix_x']), int(left['pix_y'])]
                    end_r = [int(right['pix_x']), int(right['pix_y'])]

                    p1 = self.get_point(end_l, p0)
                    p2 = self.get_point(end_r, p0)

                    cv2.line(mask, pt1=tuple(p0), pt2=tuple(p1), color=(255, 255, 255), thickness=1)
                    cv2.line(mask, pt1=tuple(p0), pt2=tuple(p2), color=(255, 255, 255), thickness=1)

        return ig_data, mask/255.0

    def get_point(self, end_l, p0):
        p1 = []
        length = 5
        num = max([abs(p0[0] - end_l[0]), abs(p0[1] - end_l[1])])
        st = min([p0[0], end_l[0]])
        ed = max([p0[0], end_l[0]])
        num = max(6, num)

        xl = np.linspace(start=st, stop=ed, num=num)
        if st == p0[0]:
            p1.append(int(xl[length]))
        else:
            p1.append(int(xl[-length]))
        st = min([p0[1], end_l[1]])
        ed = max([p0[1], end_l[1]])
        yl = np.linspace(start=st, stop=ed, num=num)
        if st == p0[1]:
            p1.append(int(yl[length]))
        else:
            p1.append(int(yl[-length]))
        return p1

def gen_pix2pix():
    direct = np.zeros(shape=[256, 512, 3], dtype=np.uint8)
    d = BigLand([256, 256])
    for x in range(d.len()):
        img, mask = d.pull_item(x)
        direct[:,0:256,:] = img[:,:,0:3]
        direct[:, 256:, :] = mask
        io.imsave('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/pix2pix/'+str(x)+'.jpg',direct)

def t_area():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/'
    image_size = [256, 256]
    data_set = BigLandMask(image_size=image_size)
    ttlb = []
    print(data_set.len())
    index = range(data_set.len())
    index = np.asarray(index)

    for x in index:

        result = data_set.pull_item(x)


def tt():
    d = BigLandMaskCoc([256,256])
    for x in range(100):
        ig_data, mask = d.pull_item(x)

        plt.imshow(ig_data)
        plt.show()
        plt.imshow(mask[:,:,0])
        plt.show()



