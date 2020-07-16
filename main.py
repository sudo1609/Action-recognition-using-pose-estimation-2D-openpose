import cv2 
import numpy as np 
import argparse
import time
from utils import choose_run_mode, load_pretraind_openpose
from pose.Detected_keypoints import detect_keypoints
from pose.pose_visualizer import visualizer, getPersonwiseKeypoints, getValidPairs
from action.Recognizer import action
from pose.data_preprocessing import pose_normalization

parser = argparse.ArgumentParser("action recognition by openpose")
parser.add_argument("--video", help='path to video file')
args = parser.parse_args()

#load models openpose
net = load_pretraind_openpose()


realtime_fps = 0
start = time.time()
frame_count = 0
fps_interval = 1
fps_count = 0


#choose mode webcam or video
cap = choose_run_mode(args)


frameHeight = 1440
frameWidth = 1440
imHeight = 368
imWidth = int((imHeight/frameHeight) * frameWidth)

while cv2.waitKey(1) < 0:
    hasFame, image = cap.read()
    if hasFame:
        fps_count += 1
        frame_count += 1
    
    in_blob = cv2.dnn.blobFromImage(image, 1.0/255,
                (imWidth, imHeight), (0,0,0), swapRB = False, crop = False)
    net.setInput(in_blob)
    output = net.forward()

    get_keypoints, keypoint_list = detect_keypoints(image, output)
    valid_pairs, invalid_pairs = getValidPairs(image, output, get_keypoints, keypoint_list)
    
    image = visualizer(image, get_keypoints, keypoint_list, output)
    #action:
    image = action(image, valid_pairs, invalid_pairs, keypoint_list, get_keypoints)

    #Đo số FPS
    if (time.time() - start) > fps_interval:
        realtime_fps = fps_count / (time.time() - start)
        fps_count = 0
        start = time.time()
    fps_label = "FPS:{0:0.2f}".format(realtime_fps)
    cv2.putText(image, fps_label, (image.shape[1] - 150 , 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    #Tổng thời gian chạy và tổng số khung hình
    if frame_count == 1:
        run = time.time()
    run_timer = time.time() - run
    time_frame_lable = "{0:.2f} | {1}".format(run_timer, frame_count)
    cv2.putText(image, time_frame_lable, (image.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('recognition', image)
    
    #collect data: 
    """personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoint_list)
    coordi = list()
    for i in range(len(personwiseKeypoints[0]) - 1):
        check = personwiseKeypoints[0][i]
        if check == -1:
            coordi.append((0.0,0.0))
        elif check == get_keypoints[i][0][-1]:
            coordi.append(get_keypoints[i][0][0:2])

    skeleton = pose_normalization(coordi)  
    with open("data_action.txt", 'a') as f:
        f.write(str(skeleton).lstrip('[').rstrip(']'))
        f.write(', wave')
        f.write('\n')
    print(skeleton)
    print(np.array(skeleton).shape)"""




