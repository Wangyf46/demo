import numpy as np
import cv2
import time
import os
import ipdb


# /data/wangyf/test_videos/v2.mp4  (756, 630). (1388, 630) ok
# v1.map4 

class BBox_Label(object):
    def __init__(self, ID, x_location, y_location, flag = 0):
        self.ID = ID
        self.x_location = x_location
        self.y_location = y_location
        self.flag = flag

class Frame_Label(object):
    def __init__(self, bboxs_len, bboxs = []):
        self.bboxs_len = bboxs_len
        self.bboxs = bboxs

# pre_frame = None
# count = 0
def crossline_check_count(frame, line, tracker_info, 
                          count, pre_frame):
    cv2.line(frame, (line[0], line[1]), (line[2], line[1]), (0, 255, 0), 6)
    cv2.nameWindow("show1", 0)
    cv2.imshow("show1", frame)
    if cv2.waitKey(1) & 0XFF == ord(' '):
        cv2.waitKey(0)
    bboxs = []
    for i in range(len(tracker_info)):
        ID, x, y = (tracker_infox[i][0], tracke_info[i][1], tracker_info[i][2])
        if x >= line[0] and x <= line[2]:
            if y <= line[1]:
                flag = -1
            else:
                flag = 1
        bbox = BBox_Label(ID, x, y, flag)
        bboxs.append(bbox)
    cur_frame = Frame_Label(len(tracker_info), bboxs)
    if pre_frame is not None and cur_frame is not None:
        cur_bboxs_len = cur_frame.bboxs_len
        pre_bboxs_len = pre_frame.bboxs_len
        for p in range(cur_bboxs_len):
            for q in range(pre_bboxs_len):
                if cur_frame.bboxs[p].ID == pre_frame.bboxs[q].ID:
                    if cur_frame.bboxs[p].flag == 1 and pre_frame.bboxs[q].flag == -1:
                        count += count
                        print(count)
    pre_frame = cur_frame
    
    return pre_frame, count
                

