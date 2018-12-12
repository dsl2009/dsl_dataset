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
class Tree(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr,'*','*.png'))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),

        ])

    def len(self):
        return len(self.images)

    def pull_item(self,index):
        image_path = self.images[index]
        json_path = image_path.replace('.png','.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]
        total = len(bundry)
        msk = np.zeros((shape[0], shape[1], total))
        ids = []

        for idx, b in enumerate(bundry):
            mask = np.zeros(shape=(shape[0], shape[1], 3))
            pts = []
            for p in b['boundary']:
                pts.append([int(p['x']), int(p['y'])])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))

            msk[:, :, idx] = mask[:, :, 0]
            ids.append(0)

        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig, self.image_size)
        msk = utils.resize_mask(msk,scale,padding,crop)
        if random.randint(0,1) ==1:
            ag = self.aug.to_deterministic()
            ig = ag.augment_image(ig)
            msk = ag.augment_image(msk)

        box = utils.extract_bboxes(msk)
        ids = np.asarray(ids)
        mj = (box[:,3] -box[:,1])*(box[:,2] -box[:,0])
        mk = np.where(mj<self.image_size[0]*self.image_size[1]/3/3)
        box = box[mk]
        ids = ids[mk]

        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj >0.5)
        box = box[mk]
        ids = ids[mk]

        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])

        mk = np.where(mj < 2)
        box = box[mk]
        ids = ids[mk]

        box = box / np.asarray([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])
        return ig, box,ids

class Tree_mask(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr,'*','*.png'))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),

        ])

    def len(self):
        return len(self.images)

    def pull_item(self,index):
        image_path = self.images[index]
        json_path = image_path.replace('.png','.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]
        orchard = []
        msk = []
        ids = []
        for idx, b in enumerate(bundry):
            if b['correction_type'] == 'tree':
                mask = np.zeros(shape=(shape[0], shape[1], 3))
                pts = []
                for p in b['boundary']:
                    pts.append([int(p['x']), int(p['y'])])
                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], color=(255, 255, 255))

                msk.append(mask[:, :, 0:1])
                ids.append(0)
            elif b['correction_type'] == 'orchard':
                mask = np.zeros(shape=(shape[0], shape[1], 3))
                pts = []
                for p in b['boundary']:
                    pts.append([int(p['x']), int(p['y'])])
                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], color=(255, 255, 255))

                orchard.append(mask[:, :, 0:1])


        if len(msk)>1:
            msk = np.concatenate(msk,axis=2)
        elif len(msk) ==1:
            msk = msk[0]
        else:
            return None

        if len(orchard)>1:
            orchard = np.concatenate(orchard,axis=2)
        elif len(orchard) ==1:
            orchard = orchard[0]
        else:
            return None


        orchard = np.expand_dims(np.sum(orchard, axis=2), -1)






        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig, self.image_size)
        msk = utils.resize_mask(msk,scale,padding,crop)
        orchard = utils.resize_mask(orchard,scale,padding,crop)
        if random.randint(0,1) ==1:
            ag = self.aug.to_deterministic()
            ig = ag.augment_image(ig)
            msk = ag.augment_image(msk)
            orchard = ag.augment_image(orchard)

        box = utils.extract_bboxes(msk)
        ids = np.asarray(ids)
        mask = np.transpose(msk, [2,0,1])


        mj = (box[:,3] -box[:,1])*(box[:,2] -box[:,0])


        mk = np.where(mj<self.image_size[0]*self.image_size[1]/3/3)
        box = box[mk]
        ids = ids[mk]
        mask = mask[mk]

        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj >0.4)
        box = box[mk]
        ids = ids[mk]
        mask = mask[mk]


        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj < 2.5)


        box = box[mk]
        ids = ids[mk]
        mask = mask[mk]


        mask = np.transpose(mask, [1, 2, 0])
        #mask = utils.minimize_mask(box, mask, mini_shape=(28, 28))
        mask = np.sum(mask, 2)
        mask[np.where(mask>255)] = 255
        box = box / np.asarray([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])
        ig = ig*(orchard/255).astype(np.uint8)
        return ig, box,ids,mask


class Tree_edge(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr, '*', '*.png'))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),
        ])

    def len(self):
        return len(self.images)

    def pull_item(self,index):
        image_path = self.images[index]
        json_path = image_path.replace('.png', '.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]
        total = len(bundry)
        msk = np.zeros((shape[0], shape[1], total))
        ids = []

        for idx, b in enumerate(bundry):
            mask = np.zeros(shape=(shape[0], shape[1], 3))
            pts = []
            for p in b['boundary']:
                pts.append([int(p['x']), int(p['y'])])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))

            msk[:, :, idx] = mask[:, :, 0]
        msk = np.sum(msk,2)
        msk[np.where(msk > 0)] = 255
        if random.randint(0, 1) == 1:
            ag = self.aug.to_deterministic()
            ig = ag.augment_image(ig)
            msk = ag.augment_image(msk)
        return ig, msk

class Tree_edge_loc(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr, '*', '*.png'))


    def len(self):
        return len(self.images)

    def pull_item(self,index):
        image_path = self.images[index]
        json_path = image_path.replace('.png', '.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]
        orchard = []
        msk = []
        ids = []
        mask = np.zeros(shape=(shape[0], shape[1], 5), dtype=np.float32)
        for idx, b in enumerate(bundry):
            if b['correction_type'] == 'tree':
                boun = b['boundary'][0:-1]
                for ix, p in enumerate(boun):
                    c = np.zeros(shape=(shape[0], shape[1], 3), dtype=np.uint8)
                    cv2.circle(c, center=(int(p['x']), int(p['y'])),radius=3, color=(255,255,255),thickness=-1)

                    #mask[int(p['x']), int(p['y']), 0] = 1.0
                    mask[:,:,0] = mask[:,:,0]+c[:,:,0]/255.0
                    left = boun[ix-1]
                    if ix== len(boun)-1:
                        right = boun[0]
                    else:
                        right = boun[ix+1]




                    mask[int(p['x']), int(p['y']), 1] = float(left['x'])/self.image_size[0]
                    mask[int(p['x']), int(p['y']), 2] = float(left['y']) / self.image_size[1]

                    mask[int(p['x']), int(p['y']), 3] = float(right['x']) / self.image_size[0]
                    mask[int(p['x']), int(p['y']), 4] = float(right['y']) / self.image_size[1]


        '''
        plt.subplot(221)
        plt.imshow( mask[:,:,0])
        plt.subplot(222)
        plt.imshow(mask[:, :, 1])

        plt.subplot(223)
        plt.imshow(mask[:, :, 2])
        plt.subplot(224)
        plt.imshow(mask[:, :, 3])

        plt.show()
        '''



        return ig, mask


class Tree_edge_coc(object):
    def __init__(self, root_dr, image_size, mask_pool_size=28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr, '*', '*.png'))

    def len(self):
        return len(self.images)

    def pull_item(self, index):
        image_path = self.images[index]
        json_path = image_path.replace('.png', '.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]
        orchard = []
        msk = []
        ids = []
        mask = np.zeros(shape=(shape[0], shape[1], 3), dtype=np.uint8)
        for idx, b in enumerate(bundry):
            if b['correction_type'] == 'tree':
                boun = b['boundary'][0:-1]

                for ix, p in enumerate(boun):
                    p0 = [int(p['x']), int(p['y'])]

                    left = boun[ix - 1]
                    if ix == len(boun) - 1:
                        right = boun[0]
                    else:
                        right = boun[ix + 1]

                    end_l = [int(left['x']), int(left['y'])]
                    end_r = [int(right['x']), int(right['y'])]

                    p1 = self.get_point(end_l, p0)
                    p2 = self.get_point(end_r, p0)

                    cv2.line(mask, pt1=tuple(p0),pt2=tuple(p1),color=(255,0,0),thickness=1)
                    cv2.line(mask, pt1=tuple(p0), pt2=tuple(p2), color=(255, 0, 0), thickness=1)



        plt.imshow(mask)

        plt.show()

        return ig, mask

    def get_point(self, end_l, p0):
        p1 = []
        num = max([abs(p0[0] - end_l[0]), abs(p0[1] - end_l[1])])
        st = min([p0[0], end_l[0]])
        ed = max([p0[0], end_l[0]])
        num = max(6,num )

        xl = np.linspace(start=st, stop=ed, num=num)
        if st == p0[0]:
            p1.append(int(xl[2]))
        else:
            p1.append(int(xl[-2]))
        st = min([p0[1], end_l[1]])
        ed = max([p0[1], end_l[1]])
        yl = np.linspace(start=st, stop=ed, num=num)
        if st == p0[1]:
            p1.append(int(yl[2]))
        else:
            p1.append(int(yl[-2]))
        return p1


def tt():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/'
    image_size = [256, 256]
    data_set = Tree_mask(image_dr,  image_size=image_size)
    ttlb = []
    print(data_set.len())
    index = range(data_set.len())
    index = np.asarray(index)

    for x in index:

        result = data_set.pull_item(x)

        if result:
            ig, box, ids, mask = data_set.pull_item(x)
            plt.subplot(121)
            plt.imshow(mask)
            plt.subplot(122)
            plt.imshow(ig)
            plt.show()

            if ids is not None and len(ids) > 0:
                box = box * np.asarray([image_size[1], image_size[0],image_size[1], image_size[0]])
                mj = (box[:,2]-box[:,0])*(box[:,3]-box[:,1])
                print(mj)
                print(np.sqrt(mj))

                visual.display_instances(ig, box)
def tt1():
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/'
    image_size = [256, 256]
    data_set = Tree_edge_coc(image_dr,  image_size=image_size)
    ttlb = []

    index = range(data_set.len())
    index = np.asarray(index)
    np.random.shuffle(index)
    for x in index:

        result = data_set.pull_item(x)

        if result:
            ig, mask = data_set.pull_item(x)
