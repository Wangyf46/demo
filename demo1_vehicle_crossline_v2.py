import numpy as np
import cv2
import time
import os
import ipdb
import glob
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
from PIL import Image
from PIL import ImageDraw
from src import utils

class BBox_Label(object):
    def __init__(self, name, percentage_probability, box_points,
                 x_location, y_location, ID = None, flag = 0):
        self.name = name
        self.percentage_probability = percentage_probability
        self.box_points = box_points
        self.x_location = int(x_location)
        self.y_location = int(y_location)
        self.ID = ID
        self.flag = flag


class Frame_Label(object):
    def __init__(self, frame_num, bbox_num, bboxs = []):
        self.frame_num = frame_num
        self.bbox_num = bbox_num
        self.bboxs = bboxs

class Ed_Index_label(object):
    def __init__(self,index_i, index_j, Euclidean_Distance):
        self.index_i = index_i
        self.index_j = index_j
        self.Euclidean_Distance = Euclidean_Distance

def vehicle_crossline( video1, l1):
    cap = cv2.VideoCapture(video1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps {0}, size {1}".format(fps, size))

    ## output save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video =  cv2.VideoWriter('demo1.avi', fourcc, fps, size)

    ret, frame = cap.read()
    execution_path = os.getcwd()
    detector = ObjectDetection()
    #detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path ,
                                       "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed = "normal")

    #video_path  = detector.detectObjectsFromVideo(              
                                   # input_file_path = root + video1,
                                   # output_file_path = root + "detected",
                                   # frames_per_second = 20, log_progress=True)
    #print(video_path)

    num = 0
    car_count = 0
    pre_frame = None
    cur_frame = None
    threshold = 300
    ID_num = 0
    while(ret):
        if num % 10 ==  0:
            #cv2.line(frame, (756, 630), (1388, 630), (0, 255, 0), 6)
            cv2.line(frame, (350, 130), (800, 130), (0, 255, 0), 6)
            #cv2.rectangle(frame, (600, 1000), (1600, 300), (255, 0, 0), 4)
       
            savepath = os.getcwd() + '/' + str(num) + '.jpg'
            cv2.imwrite(savepath, frame)
            
            custom_objects = detector.CustomObjects(person=True)
            detections = detector.detectCustomObjectsFromImage(custom_objects,
                                                    input_image = savepath,
                                                    output_image_path = savepath)
            bboxs = []
            for eachObject in detections:
                x = (eachObject["box_points"][0] + 
                     eachObject["box_points"][2]) / 2
                y = (eachObject["box_points"][1] + 
                     eachObject["box_points"][3]) / 2
                ID = ID_num
                ID_num += 1
                if x >= 350 and x <= 800:
                    if y  <= 130:
                        flag = -1
                    else:
                        flag = 1
                else:
                    flag = 0
                bbox = BBox_Label(eachObject["name"], 
                                  eachObject["percentage_probability"], 
                                  eachObject["box_points"],
                                  x, y, ID, flag)
                bboxs.append(bbox)
                '''
                p1 = (eachObject["box_points"][0], eachObject["box_points"][3])
                p2 = (eachObject["box_points"][2], eachObject["box_points"][3])
                flags = utils.intersect(l1, p1, p2)
                x = eachObject["box_points"][2] - eachObject["box_points"][0]
                y = eachObject["box_points"][3] - eachObject["box_points"][1]
                if flags == True:
                    total_car += 1
                    print(("f_num: %d; car_ID:%f; total_car: %d") %
                         (num, eac hObject["percentage_probability"], total_car))
                '''
            cur_frame = Frame_Label(num, len(detections), bboxs)
            if pre_frame is not None and cur_frame is not None:
                cur_bbox_num = cur_frame.bbox_num
                pre_bbox_num = pre_frame.bbox_num
                for i in range(cur_bbox_num):
                    eu_ds = []
                    min_eu_d = None
                    for j in range(pre_bbox_num):
                        diff_x = (cur_frame.bboxs[i].x_location - 
                                 pre_frame.bboxs[j].x_location)
                        diff_y = (cur_frame.bboxs[i].y_location - 
                                  pre_frame.bboxs[j].y_location)
                        Euclidean_Distance = np.sqrt(diff_x ** 2 + diff_y ** 2)
                        if Euclidean_Distance <= threshold:
                            eu_d = Ed_Index_label(i, j, Euclidean_Distance) 
                            eu_ds.append(eu_d)
                    if len(eu_ds) != 0:
                        min_eu_d = eu_ds[0]
                        for m in range(1, len(eu_ds)):
                            if (min_eu_d.Euclidean_Distance > eu_ds[m].Euclidean_Distance):
                                min_eu_d = eu_ds[m]
                        cur_frame.bboxs[i].ID = pre_frame.bboxs[min_eu_d.index_j].ID
                for p in range(cur_bbox_num):
                    for q in range(pre_bbox_num):
                        if cur_frame.bboxs[p].ID == pre_frame.bboxs[q].ID:
                            if cur_frame.bboxs[p].flag == 1 and pre_frame.bboxs[q].flag == -1:
                                car_count += 1 # TODO
                                print("the car of cross line count is %d" % car_count)
            dst = cv2.imread(savepath) 
            font = cv2.FONT_HERSHEY_SIMPLEX
           # for n in range(len(detections)):
              # # print(cur_frame.bboxs[n].ID, cur_frame.bboxs[n].percentage_probability)
               # dst = cv2.putText(dst, str(cur_frame.bboxs[n].ID),
                #                 (cur_frame.bboxs[n].x_location, cur_frame.bboxs[n].y_location),
                 #                 font, 1, (255, 0, 0), 2)
            dst = cv2.putText(dst, str(car_count), (50, 50), font , 2, (0, 255, 0), 4)
            cv2.namedWindow("show1", 0)
            cv2.resizeWindow("show1", 1000, 700)
            cv2.imshow("show1", dst)
            if cv2.waitKey(1) & 0XFF == ord(' '):
                cv2.waitKey(0) 

            cv2.imwrite(savepath, dst)
            output_video.write(dst)
            
            pre_frame = cur_frame
        ret, frame = cap.read()
        num += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # root = "/nfs-data/wangyf/131/data/test_videos/car/"
    # video1 = "v2.mp4"
    video1 = '/nfs-data/pku_20190601/2/1.mp4'
    #l1 = (0, 829, 1600, 909)
    l1 = (600, 720, 1560, 710)
    vehicle_crossline(video1, l1)

'''
v1.mp4:  cv2.line(frame, (600, 630), (1570, 630), (0, 255, 0), 6)    630
v2.mp4:  cv2.line(frame, (756, 630), (1388, 630), (0, 255, 0), 6)    630   OK
v3.mp4:
'''
