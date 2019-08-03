'''
functions: vehicle crossline count
model:     ImageAi ObjectDections
env:       Python 3.7|tensorflow
'''

import numpy as np
import cv2
import time
import os
from imageai.Detection import ObjectDetection
from PIL import Image
from PIL import ImageDraw
from src import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
s1 = time.time()
root = "/data2/crowd_task/"
video1 = "demo2.avi"
video2 = "demo2_out.avi"
cap = cv2.VideoCapture(root + video1)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps {0}, size {1}".format(fps, size))
ret, frame = cap.read()
#fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#cap_out = cv2.VideoWriter(root + video2, fourcc, fps, size)

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed = "faster")

l1 = (0, 829, 1600, 909)
num = 0
total_car = 0
while(ret):
    t1 = time.time()
    if num <= 1000:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.line(l1, fill = "green", width = 6)
        #img.show()
        img = np.array(img)
        cv2.imwrite(root + "dst.jpg", img)
        custom_objects = detector.CustomObjects(car = True)
        detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                              input_image = os.path.join(root, "dst.jpg"),
                                              output_image_path = os.path.join(root, "dst.jpg"))
        t2 = time.time()
        dst = cv2.imread(os.path.join(root, "dst.jpg"))
        cv2.namedWindow("videoshow", 0)
        cv2.resizeWindow("videoshow", 1200, 1200)
        cv2.imshow("videoshow", dst)
        if cv2.waitKey(1) & 0XFF == ord(' '):
            cv2.waitKey(0)
        i = 0
        for eachObject in detections:
            p1 = (eachObject["box_points"][0], eachObject["box_points"][3])
            p2 = (eachObject["box_points"][2], eachObject["box_points"][3])
            flags1 = utils.intersect(l1, p1, p2)
            if flags1 == True:
                total_car += 1
            if flags1 == True :
                print(("f_num: %d; car_ID:%f; total_car: %d") %
                        (num, eachObject["percentage_probability"], total_car))
        t3 = time.time()
        #print(round(t2-t1,2), round(t3-t2,2))
        ret, frame = cap.read()
        num += 1
    else:
        break
cap.release()