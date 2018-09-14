from pycocotools.coco import COCO
import os
from dsl_data.coco_handler import annToMask
from matplotlib import pyplot as plt
from skimage import io
from skimage import measure
import random
import numpy as np
import cv2
coco_image_dir = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/building/train/images'
ann = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/building/train/annotation.json'


def gen_cnt(mask):
    mask = mask*255
    mask= mask.astype(np.uint8)
    io.imsave('tmp.jpg',mask)
    im = cv2.imread('tmp.jpg',0)
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ig = np.zeros([mask.shape[0],mask.shape[1],3],np.uint8)
    cv2.drawContours(ig,contours,-1,(255,255,255),thickness=2)

    return ig[:,:,0]
def get_data(idx,image_size,coco,map_source_class_id):
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[idx], catIds=100, iscrowd=False))
    img_url = os.path.join(coco_image_dir, coco.imgs[idx]["file_name"])
    instance_masks = []
    cls_ids = []
    cnt = []
    for annotation in annotations:
        class_id = map_source_class_id[annotation['category_id']]

        m = annToMask(annotation, coco.imgs[idx]["height"],
                      coco.imgs[idx]["width"])
        m = cv2.resize(m, dsize=(image_size, image_size))
        ct = gen_cnt(m)
        cnt.append(ct)
        instance_masks.append(m)
        cls_ids.append(class_id)
    img = io.imread(img_url)
    img = cv2.resize(img,dsize=(image_size, image_size))
    mask = np.asarray(instance_masks)
    mask = np.sum(mask,axis=0)
    cts = np.sum(np.asarray(cnt),axis=0)
    cts = cts.astype(np.float32)/255.0
    cts = np.clip(cts,0,1)
    mask = mask.astype(np.float32)
    return img,mask,cts



def get_building(batch_size,is_shuff = True,image_size=512):
    coco = COCO(ann)
    class_ids = sorted(coco.getCatIds())
    image_ids = list(coco.imgs.keys())
    map_source_class_id = dict(zip(class_ids, range(len(class_ids))))
    length = len(image_ids)
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img, mask, ct = get_data(image_ids[idx[index]],image_size,coco,map_source_class_id)
            except:
                index = index+1
                print(index)
                continue
            img = img - [123.15, 115.90, 103.06]
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                masks = np.zeros(shape=[batch_size, image_size, image_size], dtype=np.int)
                cts = np.zeros(shape=[batch_size,image_size,image_size],dtype=np.float32)

                images[b,:,:,:] = img
                masks[b,:,:] = mask
                cts[b,:,:] = ct

                b=b+1
                index = index + 1

            else:
                images[b, :, :, :] = img
                masks[b, :, :] = mask
                cts[b, :, :] = ct
                b = b + 1
                index = index + 1

            if b>=batch_size:
                yield [images,masks,cts]
                b = 0

            if index>= length:
                index = 0

