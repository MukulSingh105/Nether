import cv2
from collections import OrderedDict
import numpy as np


# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
def get_frames_in_video(video_file):
    time_to_frame = {} #$OrderedDict()
    vid = cv2.VideoCapture(video_file) # https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
    success,image = vid.read()
    count = 0
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(fps)
    curr_frame = 0
    while success and (count < 15): # want only the 1st 15 frames
        success,image = vid.read()
        if (curr_frame % fps) == 0: 
            cv2.imwrite("frame%d.jpg" %count, image)
            time_to_frame[count] = vid.get(cv2.CAP_PROP_POS_MSEC)
            print("Read a new frame %d: "%count, success)
            count += 1
        curr_frame += 1
    return time_to_frame

# TODO
def compare_frame_set():
    pass

def zoom(video_file):
    time_to_frame = get_frames_in_video(video_file)
    for key1 in time_to_frame:
        img1 = cv2.cvtColor(cv2.imread("frame"+str(key1)+".jpg"), cv2.COLOR_BGR2GRAY)
        for key2 in time_to_frame:
            if key1 == key2:
                continue
            img2 = cv2.cvtColor(cv2.imread("frame"+str(key2)+".jpg"), cv2.COLOR_BGR2GRAY)
            error = mse(img1, img2)
            print("error between %d and %d is %f" %(key1, key2, error))

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

    # # splotlight the person who's talking by measuring facial landmark delta

    # # ONLY works for videos
    # pass

zoom("input_video.mp4")