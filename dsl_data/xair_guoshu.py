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
class Tree(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr,'*','*.png'))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
        ])
    def len(self):
        return len(self.images)

    def pull_item(self,index):
        image_path = self.images[index]
        json_path = image_path.replace('.png','.json')
        bundry = json.loads(open(json_path).read().encode('utf8'))
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        ig = aug_utils.pytorch_aug_color(ig)
        shape = ig.shape[0:2]
        total = len(bundry)
        msk = np.zeros((shape[0], shape[1], total),dtype=np.uint8)
        ids = []
        orchard = np.zeros(shape=(shape[0], shape[1], 3),dtype=np.uint8)
        msk = []

        for idx, b in enumerate(bundry):
            if b['correction_type'] == 'tree':
                mask = np.zeros(shape=(shape[0], shape[1], 3),dtype=np.uint8)
                pts = []
                for p in b['boundary']:
                    pts.append([int(p['x']), int(p['y'])])
                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], color=(255, 255, 255))

                msk.append(mask[:, :, 0:1])
                ids.append(0)
            elif b['correction_type'] == 'orchard':
                mask = np.zeros(shape=(shape[0], shape[1], 3),dtype=np.uint8)
                pts = []
                for p in b['boundary']:
                    pts.append([int(p['x']), int(p['y'])])
                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], color=(255, 255, 255))

                orchard+=mask
        ig = ig*(orchard/255)


        if len(msk) > 1:
            msk = np.concatenate(msk, axis=2)
        elif len(msk) == 1:
            msk = msk[0]
        else:
            return None


        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig, self.image_size)
        msk = utils.resize_mask(msk,scale,padding,crop)
        if random.randint(0,1) !=1:
            ag = self.aug.to_deterministic()
            ig = ag.augment_image(ig)
            msk = ag.augment_image(msk)

        box = utils.extract_bboxes(msk)
        ids = np.asarray(ids)

        


        mj = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
        mk = np.where(mj > self.image_size[0] * self.image_size[1] / 32 / 32)
        box = box[mk]
        ids = ids[mk]


        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj >0.25)
        box = box[mk]
        ids = ids[mk]

        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj < 4)
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

        mj = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
        mk = np.where(mj > self.image_size[0] * self.image_size[1] / 32 / 32)
        box = box[mk]
        ids = ids[mk]
        mask = mask[mk]

        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj >0.3)
        box = box[mk]
        ids = ids[mk]
        mask = mask[mk]


        mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
        mk = np.where(mj < 3)
        box = box[mk]
        ids = ids[mk]
        mask = mask[mk]

        orchard = np.sum(orchard, axis=2)
        mask = np.transpose(mask, [1, 2, 0])
        #mask = utils.minimize_mask(box, mask, mini_shape=(28, 28))
        mask = np.sum(mask, 2)
        mask[np.where(mask>255)] = 255
        box = box / np.asarray([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])
        #ig = ig*(orchard/255).astype(np.uint8)
        return ig, box,ids,mask


class Tree_mask_ins(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28, output_size=[128,128]):
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
        self.factor = iaa.Sequential([
            iaa.Scale(size=output_size),
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
                mask = np.zeros(shape=(shape[0], shape[1], 3),dtype=np.uint8)
                pts = []
                for p in b['boundary']:
                    pts.append([int(p['x']), int(p['y'])])
                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], color=(255, 255, 255))

                msk.append(mask[:, :, 0:1])
                ids.append(0)
            elif b['correction_type'] == 'orchard':
                mask = np.zeros(shape=(shape[0], shape[1], 3),dtype=np.uint8)
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



        #ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig, self.image_size)
        #msk = utils.resize_mask(msk,scale,padding,crop)
        #orchard = utils.resize_mask(orchard,scale,padding,crop)
        if random.randint(0,1) ==1:
            ag = self.aug.to_deterministic()
            ig = ag.augment_image(ig)
            msk = ag.augment_image(msk)

        instance_masks = self.factor.augment_image(msk)
        instance_masks[np.where(instance_masks >= 128)] = 255
        instance_masks[np.where(instance_masks < 128)] = 0

        seg_mask = np.sum(instance_masks, axis=2)
        seg_mask[np.where(seg_mask > 255)] = 255

        # remove too small
        d = np.sum(instance_masks, axis=(0, 1))
        k = np.where(d > 256 * 10)

        instance_masks = instance_masks[:, :, k]
        instance_masks = np.squeeze(instance_masks, axis=2)

        return ig, instance_masks, seg_mask, instance_masks.shape[2]











def get_tree_msg():
    from sklearn.cluster import KMeans
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/'
    image_size = [256, 256]
    data_set = Tree(image_dr, image_size=image_size)
    d = []

    index = range(data_set.len())
    mx = 0
    mn = 256
    t = 0
    for x in index:
        try:
            ig, box, ids = data_set.pull_item(x)
            mj = np.sqrt((box[:,2] - box[:,0])*(box[:,3] - box[:,1]))*256

            if len(mj)>0:
                d.extend(mj)

        except:
            pass
        if len(d)>100:

            cl = KMeans(n_clusters=12)
            cl.fit(np.reshape(np.asarray(d),(-1,1)))
            print(cl.cluster_centers_)
    cl = KMeans(n_clusters=12)
    cl.fit(np.asarray(d))
    print(cl.cluster_centers_)

if __name__ == '__main__':
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/'
    image_size = [256, 256]
    data_set = Tree(image_dr, image_size=image_size)
    '''
        for i in range(100):
        try:
            ig, box, ids = data_set.pull_item(i)
            visual.display_instances(ig, box)
        except:
            pass
    '''



