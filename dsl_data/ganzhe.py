import cv2
import json
import numpy as np
ig = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/tt/0101_107.jpg'
igs = cv2.imread(ig)
a = json.loads(open(ig.replace('.jpg','.json')).read())
for x in a['shapes']:
    cv2.fillPoly(igs, np.asarray([x['points']], np.int), (0, 0, 255))
cv2.imwrite('s.jpg', igs)