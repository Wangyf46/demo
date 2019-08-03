import os
import cv2
import glob
import ipdb

img_path = '/data/wangyf/demo/demo1/'
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'XVID')
VW1 = cv2.VideoWriter('demo1.avi', fourcc, fps, (1920, 1080))

img_files = sorted(glob.glob(os.path.join(img_path,'*.jpg')))
ims = [cv2.imread(imf) for imf in img_files]

for f, im in enumerate(ims):
    print(im.shape)
    VW1.write(im)
VW1.release()
