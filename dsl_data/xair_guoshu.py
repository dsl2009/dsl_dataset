import glob
import os
import cv2
import numpy as np
import json
from dsl_data import utils
from dsl_data import visual
class Tree(object):
    def __init__(self, root_dr, image_size,mask_pool_size = 28):
        self.root_dr = root_dr
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.images = glob.glob(os.path.join(root_dr,'*.jpg'))

    def len(self):
        return len(self.images)

    def pull_item(self,idx):
        image_path = self.images[idx]
        json_path = image_path.replace('.jpg','.json')
        bundry = json.loads(open(json_path).read())
        ig = cv2.imread(image_path)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]
        total = len(bundry)
        msk = np.zeros((shape[0], shape[1], total))
        ids = []

        for idx, b in enumerate(bundry):
            mask = np.zeros(shape=(shape[0], shape[1], 3))
            pts = []
            for p in b:
                pts.append([int(p['pix_x']), int(p['pix_y'])])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))
            msk[:, :, idx] = mask[:, :, 0]
            ids.append(0)
        ig, window, scale, padding, crop = utils.resize_image_fixed_size(ig, self.image_size)
        msk = utils.resize_mask(msk,scale,padding,crop)
        box = utils.extract_bboxes(msk)
        mask = utils.minimize_mask(box, msk, mini_shape=(self.mask_pool_size, self.mask_pool_size))
        box = box / np.asarray([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])
        return ig, box,ids

def tt():
    d = Tree('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/guoshu/data',512)
    for x in range(100):
        ig, box = d.pull_item(x)
        visual.display_instances(ig, box*512)
