import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageDraw
from src.utils import *


## video_path: /data/wangyf/test_videos/5_min.avi
## area[400,400,1000,900]


 def crowd_crossline(frame, area, bboxs_crowd, bboxs_vehicle):
    x1 = area[0]
    y1 = area[1]
    x2 = area[2]
    y2 = area[3]

    l1 = (x1, y1, x2, y1)
    l2 = (x2, y1, x2, y2)
    l3 = (x2, y2, x1, y2)
    l4 = (x1, y1, x1, y2)

    
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 5)
    cv2.namedWindow("show1", 0)
    cv2.resizeWindow("show1", 1000, 700)
    cv2.imshow("show1", frame)
    if cv2.waitKey(1) & 0XFF == ord(' '):
        cv2.waitKey(0)
    
    
    
    crowd_num = 0
    for i in range(len(bboxs_crowd)):
        p1 = bboxs_crowd[i][0]
        q1 = bboxs_crowd[i][1]
        p2 = bboxs_crowd[i][2]
        q2 = bboxs_crowd[i][3]
        crowd_flag_1 = False
        if p1 >= x1 and p2 <= x2 and q1 >= y1 and q2 <= y2:
            crowd_flag_1 = True
            crowd_flag_2 = intersect(l1, (p1,q1), (p2,q2))
            crowd_flag_3 = intersect(l2, (p1,q1), (p2,q2))
            crowd_flag_4 = intersect(l3, (p1,q1), (p2,q2))
            crowd_flag_5 = intersect(l4, (p1,q1), (p2,q2))
            crowd_flag = (crowd_flag_1 or crowd_flag_2 or  crowd_flag_3
                           or crowd_flag_4 or crowd_flag_5)
            if crowd_flag == True:
                crowd_num += 1
                print("%d Crowd Intrusion, ID is %s" 
                        % (crowd_num, bboxs_crowd[i][5]))

    vehicle_num = 0
    for j in range(len(bboxs_vehicle)):
        p1 = bboxs_vehicle[i][0]
        q1 = bboxs_vehicle[i][1]
        p2 = bboxs_vehicle[i][2]
        q2 = bboxs_vehicle[i][3]
        vehicle_flag_1 = False
        if p1 >= x1 and p2 <= x2 and q1 >= y1 and q2 <= y2:
            vehicle_flag_1 = True
            vehicle_flag_2 = intersect(l1, (p1,q1), (p2,q2))
            vehicle_flag_3 = intersect(l2, (p1,q1), (p2,q2))
            vehicle_flag_4 = intersect(l3, (p1,q1), (p2,q2))
            vehicle_flag_5 = intersect(l4, (p1,q1), (p2,q2))
            vehicle_flag = (vehicle_flag_1 or vehicle_flag_2 or vehicle_flag_3
                           or vehicle_flag_4 or vehicle_flag_5)
            if vehicle_flag == True:
                vehicle_num += 1
                print("%d Vehicle Intrusion, ID is %s" 
                        % (Vehicle, bboxs_vehicle[i][5]))
    return window

