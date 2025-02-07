import cv2

# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
def get_frames_in_video(video_file):
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
            print("Read a new frame %d: "%count, success)
            count += 1
        curr_frame += 1
    return

#TODO
def compare_frame_set():
    pass

def zoom(video_file):
    get_frames_in_video(video_file)

    # compare frames

    # splotlight the person who's talking by measuring facial landmark delta

    # ONLY works for videos
    pass

get_frames_in_video("input_video.mp4")