import requests
import json
import cv2
import os
import numpy as np
import glob
import random
def down_load():
    for s in open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/guoshu/tree.json').readlines():
        data = json.loads(s.replace('\n', ''))
        urls = data['pic_url']
        name = '_'.join(urls.split('/')[-3:])
        json_name = os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/guoshu/data', name + '.json')
        name = os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/guoshu/data', name + '.jpg')
        r = requests.get(urls)
        ig = open(name, 'wb')
        ig.write(r.content)
        ig.flush()
        bun = data['boundary']

        with open(json_name,'w') as f:
            f.write(json.dumps(bun))
            f.flush()
def gen_txt(data, name):
    with open(name,'w') as f:
        for x in data:
            f.write(x+'\n')
            f.flush()


def split_train_valid():
    x = glob.glob('/home/xair/map/train/*.json')
    random.shuffle(x)
    train = x[0:-20]
    valid = x[-20:]
    gen_txt(train,'train.txt')
    gen_txt(valid,'valid.txt')

down_load()
