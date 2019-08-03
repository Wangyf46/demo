import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from src import utils

from imageai.Detection import ObjectDetection
import os


def crowd_crossline(root, video1, x1, x2, y1, y2):
    cap_in = cv2.VideoCapture(root+video1)
    fps = cap_in.get(cv2.CAP_PROP_FPS)

    size = (int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    print("fps {0}, size {1}".format(fps, size))

    ## output save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video =  cv2.VideoWriter('demo2.avi', fourcc, fps, size)

    ret, frame = cap_in.read()

    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed = "faster")

    l1 = (x1, y1, x2, y1)
    l2 = (x2, y1, x2, y2)
    l3 = (x2, y2, x1, y2)
    l4 = (x1, y1, x1, y2)
    num = 0
    while(ret):
        show = None
        if num % 15 == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            savepath = os.getcwd() + '/' + str(num) + '.jpg' 
            cv2.imwrite(savepath, frame)
            detections = detector.detectObjectsFromImage(
                    input_image = savepath,
                    output_image_path = savepath)
            dst = cv2.imread(savepath)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for eachObject in detections:
                p1 = (eachObject["box_points"][0], eachObject["box_points"][1])
                p2 = (eachObject["box_points"][2], eachObject["box_points"][1])
                p3 = (eachObject["box_points"][0], eachObject["box_points"][3])
                p4 = (eachObject["box_points"][2], eachObject["box_points"][3])
                flag_in = False
                if  (x1 <= eachObject["box_points"][0] and 
                     x2 >= eachObject["box_points"][2] and 
                     y1 <= eachObject["box_points"][1] and 
                     y2 >= eachObject["box_points"][3]):
                    flag_in = True
                flag1 = utils.intersect(l1, p1, p4)
                flag2 = utils.intersect(l2, p1, p4)
                flag3 = utils.intersect(l3, p1, p4)
                flag4 = utils.intersect(l4, p1, p4)
                flags = flag4 or flag3 or flag2 or flag1 or flag_in
                if flags == True:
                    show = 'Invade'
                    print(("f_num: %d;  Name: %s;  ID: %s;  Flag: %s") %
                         (num, eachObject["name"],
                               eachObject["percentage_probability"], 
                               flags))
            dst = cv2.putText(dst, str(show), (60, 60),font, 2, (0, 0,255), 4)
            cv2.namedWindow("videoshow1", 0)
            cv2.resizeWindow("videoshow1", 800, 600)
            cv2.imshow("videoshow1", dst)
            if cv2.waitKey(1) & 0XFF == ord(' '):
                cv2.waitKey(0)
            cv2.imwrite(savepath, dst)
            output_video.write(dst)
        ret, frame = cap_in.read()
        num += 1
    cap_in.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    root = "/nfs-data/wangyf/131/data/test_videos/"
    video1 = "5_min.avi"
   # video1 = '/data2/pku_20190601/t1/1.mp4'
    x1 = 400
    x2 = 1000
    y1 = 400
    y2 = 900
    crowd_crossline(root, video1, x1, x2, y1, y2)


