import os.path as osp
import sys
from dsl_data import utils
from imgaug import augmenters as iaa
import imgaug as ia
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
    def __init__(self, root, image_size,is_crop=True, image_sets=[ ('2007', 'trainval'),('2012', 'trainval')]):
        self.root = root
        self.image_set = image_sets
        self.image_size = image_size
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.class_mapix = dict(zip(classes, range(len(classes))))
        self.ids = list()
        self.is_crop = is_crop
        self.img_aug = iaa.Sequential([
                        iaa.Fliplr(0.5),
                        iaa.ContrastNormalization((0.7, 1.5)),
                        iaa.Multiply((0.7, 1.5)),
                        iaa.Affine(
                            scale=(0.8, 1.2)
                        )
                    ])

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
        if True:
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
def get_voc_msg():
    from sklearn.cluster import KMeans
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit'
    data_set = VOCDetection(root=image_dr, is_crop=False, image_size=[512, 512])
    ttlb = []
    image_size = [512, 512]
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
    tt()