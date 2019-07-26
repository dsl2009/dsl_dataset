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
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land_19_bk/*/*.json')
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
        edge = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)


        js_data = json.loads(open(json_pth).read())
        for b in js_data:
            label = b['correction_type']
            if label == 'land':
                points = b['boundary']
                p = []

                for pp in points:
                    p.append([pp['x'], pp['y']])
                edges = list(p)
                edges.append(p[0])
                for i in range(len(edges)-1):
                    p1 = edges[i]
                    p2 = edges[i+1]
                    if p1[0]==0 and p2[0]==0:
                        pass
                    elif p1[1]==0 and p2[1]==0:
                        pass
                    elif p1[0] == 255 and p2[0] == 255:
                        pass
                    elif p1[1] == 255 and p2[1] == 255:
                        pass
                    else:
                        cv2.line(edge,tuple(p1),tuple(p2), (255, 255, 255), thickness = 2)

                label_id = self.class_mapix[label]
                #direct = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)

                cv2.fillPoly(instance_masks, np.asarray([p], np.int), (255, 255, 255))


                # cv2.polylines(direct,np.asarray([p], np.int),True, (255,255,255), thickness=2)

        if random.randint(0,1) ==2:
            ag = self.aug.to_deterministic()
            ig_data = ag.augment_image(ig_data)
            instance_masks = ag.augment_image(instance_masks)
            edge = ag.augment_image(edge)

        return ig_data, instance_masks[:,:,0], edge[:,:,0]

class BigLandArea(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land_19_bk/*/*.json')
        self.class_mapix = dict(zip(labels,range(len(labels))))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),
        ])
        self.factor = iaa.Sequential([
            iaa.Scale(size=(128,128)),
        ])
    def len(self):
        return len(self.data)

    def draw(self,p):
        edges = list(p)
        edges.append(p[0])
        msk = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
        for i in range(len(edges) - 1):
            p1 = edges[i]
            p2 = edges[i + 1]
            if p1[0] == 0 and p2[0] == 0:
                pass
            elif p1[1] == 0 and p2[1] == 0:
                pass
            elif p1[0] == 255 and p2[0] == 255:
                pass
            elif p1[1] == 255 and p2[1] == 255:
                pass
            else:
                cv2.line(msk, tuple(p1), tuple(p2), (255, 255, 255), thickness=1)
        return msk[:,:,0:1]


    def pull_item(self,idx):
        json_pth = self.data[idx]
        image_pth = json_pth.replace('.json','.png')
        ig_data = io.imread(image_pth)[:,:,0:3]
        instance_masks = []
        js_data = json.loads(open(json_pth).read())
        for b in js_data:
            label = b['correction_type']
            if label == 'land':
                points = b['boundary']
                p = []

                for pp in points:
                    p.append([pp['x'],pp['y']])

                label_id = self.class_mapix[label]
                direct = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)

                cv2.fillPoly(direct, np.asarray([p], np.int),(255,255,255))
                #cv2.polylines(direct,np.asarray([p], np.int),True, (255,255,255), thickness=2)
                edge = self.draw(p)
                instance_masks.append(direct[:,:,0:1]-edge)


        if len(instance_masks)>1:
            instance_masks = np.concatenate(instance_masks, axis=2)
        elif len(instance_masks) ==1:
            instance_masks = instance_masks[0]
        else:
            return None, None, None, None

        if random.randint(0,1) ==2:
            ag = self.aug.to_deterministic()
            ig_data = ag.augment_image(ig_data)
            instance_masks = ag.augment_image(instance_masks)

        #scale to 64
        instance_masks = self.factor.augment_image(instance_masks)
        instance_masks[np.where(instance_masks>=128)] =255
        instance_masks[np.where(instance_masks < 128)] = 0


        seg_mask = np.sum(instance_masks, axis=2)
        seg_mask[np.where(seg_mask>255)] = 255

        #remove too small
        d = np.sum(instance_masks,axis=(0,1))
        k = np.where(d>256*10)

        instance_masks = instance_masks[:,:,k]
        instance_masks = np.squeeze(instance_masks,axis=2)


        return ig_data, instance_masks, seg_mask, instance_masks.shape[2]


class BigLandBox(object):
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
        instance_masks = np.zeros(shape=[self.image_size[0], self.image_size[1], 3])
        js_data = json.loads(open(json_pth).read())
        msk = []
        ids = []
        for b in js_data['boundary']:
            label = b['correctionType']
            if label == 'land':
                points = b['points']
                p = []
                for pp in points:
                    p.append([pp['pix_x'],pp['pix_y']])

                poly_land = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
                cv2.fillPoly(poly_land, np.asarray([p], np.int),(255,255,255))
                msk.append(poly_land[:,:,0:1])
                ids.append(0)

        if len(msk) > 1:
            msk = np.concatenate(msk, axis=2)
        elif len(msk) == 1:
            msk = msk[0]
        else:
            return None

        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig_data, self.image_size)
        msk = utils.resize_mask(msk, scale, padding, crop)

        if random.randint(0,1) ==2:
            ag = self.aug.to_deterministic()
            ig_data = ag.augment_image(ig_data)
            msk = ag.augment_image(msk)

        box = utils.extract_bboxes(msk)
        ids = np.asarray(ids)

        mask = np.sum(msk, 2)
        mask[np.where(mask > 255)] = 255
        box = box / np.asarray([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])

        return ig, box, ids, mask

class BigLandMask(object):
    def __init__(self, image_size, output_size=[256,256]):
        self.image_size = image_size
        self.output_size = output_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*/*.json')
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
        mask = np.zeros(shape=(shape[0], shape[1], 3), dtype=np.float32)
        loc_rem = np.zeros(shape=(shape[0], shape[1]), dtype=np.uint8)

        width_ratio = self.output_size[1] / self.image_size[1]
        height_ratio = self.output_size[0] / self.image_size[0]

        heatmaps = np.zeros((self.output_size[0], self.output_size[1], 3), dtype=np.float32)

        regr = []
        tags = []
        msk = []
        num_land = 0

        for b in js_data:
            num_land += 1
            label = b['correction_type']
            if label == 'land':
                boun = b['boundary'][0:-1]
                tmp_instance = []
                for ix, p in enumerate(boun):
                    tmp = np.zeros((self.output_size[0], self.output_size[1], 3), dtype=np.float32)
                    xtl, ytl = p['x'], p['y']
                    fxtl = (xtl * width_ratio)
                    fytl = (ytl * height_ratio)
                    xtl, ytl = np.clip(np.asarray([fxtl, fytl], dtype=np.int), a_min=0, a_max=self.output_size[0])
                    if True:
                        radius = 3
                        cv2.circle(tmp, center=(int(fxtl), int(fytl)), radius=radius, color=(255, 255, 255),
                                   thickness=-1)
                        tmp[np.where(heatmaps==255)]=0


                        cv2.circle(heatmaps, center=(int(fxtl), int(fytl)), radius=radius, color=(255, 255, 255),
                                   thickness=-1)
                        tmp_instance.append(tmp[:,:,0:1])
                        #utils.draw_gaussian(heatmaps, [int(fxtl), int(fytl)], radius)
                        #utils.draw_gaussian(tmp, [int(fxtl), int(fytl)], radius)
                        # regr.append([fxtl - xtl, fytl - ytl])
                        # tags.append(ytl * self.output_size[1] + xtl)
                        # msk.append(1*num_land)
                tmp_instance = np.concatenate(tmp_instance, axis=2)
                instance_masks.append(np.sum(tmp_instance, axis=2, keepdims=True))

        if len(instance_masks)>1:
            instance_masks = np.concatenate(instance_masks, axis=2)
        elif len(instance_masks) ==1:
            instance_masks = instance_masks[0]
        else:
            return None, None, None, None


        return ig_data, instance_masks,heatmaps[:,:,0], instance_masks.shape[2]

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


def draw(msk, x, y):
    import cv2
    org = np.zeros(shape=(256,256,3),dtype=np.uint8)
    start = np.asarray(list(zip(msk[1],msk[0])))
    data = (np.asarray(list(zip(x, y)))+1)/2*255
    data = data.astype(np.int32)
    for x in range(data.shape[0]):
        st = start[x]
        et = data[x]
        cv2.line(org, pt1=tuple(st), pt2=tuple(et), color=(255, 255,0),thickness=2)
    plt.imshow(org)
    plt.show()


def tt():
    d = BigLandArea(image_size=[256,256])
    for x in range(10000):
        try:
            ig_data,  instance_masks,mask,nb = d.pull_item(x)
            print(nb)
            plt.subplot(121)
            plt.imshow(ig_data)
            plt.subplot(122)
            plt.imshow(mask)
            plt.show()



        except:
            pass




if __name__ == '__main__':
    tt()


