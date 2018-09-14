import os
import glob
import shutil
from skimage import io
import json
from matplotlib import pyplot as plt
import cv2
import numpy as np
from dsl_data import utils


class AiCh(object):
    def __init__(self, dr, image_size):
        self.dr = dr
        self.image_size = image_size
        self.jsons = glob.glob(os.path.join(dr, '*.json'))

    def len(self):
        return len(self.jsons)

    def pull_item(self, idx):
        js = self.jsons[idx]
        js_data = json.loads(open(js).read())
        img_pth = os.path.join(self.dr, js_data['imagePath'])
        ig = io.imread(img_pth)
        mask = np.zeros(shape=ig.shape,dtype= np.uint8)
        for shap in js_data['shapes']:
            pts = np.array(shap['points'], np.int32)
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))
        ig, window, scale, padding, crop = utils.resize_image(ig, self.image_size, self.image_size)
        mask, window, scale, padding, crop = utils.resize_image(mask, self.image_size, self.image_size)
        return ig, mask[:,:,0]


