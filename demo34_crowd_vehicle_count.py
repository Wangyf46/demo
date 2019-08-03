import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from src.model import CSRNet
from src.utils import *
import ipdb


def crowd_count(root, video_path):
    len_window = 100
    #output_path = root + str(len_window) + '/'
    #exp_results = output_path + "exp_results.txt"
    #fd = open(exp_results, 'w')
    model = CSRNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load('multi_gpumodel_best.pth.tar') ###crowd model multi 
    #checkpoint = torch.load('s_1_100_0320model_best.pth.tar') ###vehicle model single 
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda().eval()
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps {0}, size {1}".format(fps, size))
    #index, frames = get_keyframe(len_window, cap) # key frame
    index, frames = get_frame(cap)
    FrameNum = 0
    for i in index:
        for j in range(len(frames)):
            if (i + 1) == frames[j].num:            
                cv2.namedWindow("show1", 0)
                cv2.imshow("show1", frames[j].frame)
                if cv2.waitKey(1) & 0XFF == ord(' '):
                    cv2.waitKey(0)
                frames[j].frame = cv2.resize(frames[j].frame, (500,500))
                img = Image.fromarray(frames[j].frame).convert("RGB")  #TODO RGB
                et_density_map = et_data(img, model)
                et_count = et_density_map.sum()
                et_count = int(et_count) 
                et_density_map = et_density_map * 255 / np.max(et_density_map)
                et_density_map_interpolation = get_interpolation(et_density_map) ## 
                print(et_count)
                plt.figure()                
                plt.imshow(et_density_map_interpolation, cmap = plt.cm.jet)
                plt.axis("off")
                path = str(FrameNum) + ".jpg"
                plt.savefig(path)
                # plt.show()
                # plt.close()
                
               # ipdb.set_trace()
                dst = cv2.imread(path)
                cv2.namedWindow("show2", 1)   
                cv2.imshow("show2", dst)
                # if cv2.waitKey(1) & 0XFF == ord(' '):
                cv2.waitKey(0)
        FrameNum += 1

if __name__ == '__main__':
    # root = "/nfs-data/wangyf/131/data/test_videos/"
    # video_path = root + "crowd/v2.avi"
 # video_path = root + "03_2h.avi"
    #video_path = root + "car/v2.mp4"
    root =  '/nfs-data/wangyf/'
    video_path = root + 'test2.mp4'
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,0'
    crowd_count(root, video_path)
