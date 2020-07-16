import cv2
import os
import numpy as np
import math
from pose.pose_visualizer import getPersonwiseKeypoints
from pose.data_preprocessing import pose_normalization
from keras.models import load_model


def action(img, valid_pairs, invalid_pairs, keypoints_list, get_keypoints):

    #get skeleton of each person
    def Skeleton(valid_pairs, invalid_pairs, keypoints_list, get_keypoints):
        skeleton_idx = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)
        coordi = list()
        i = 0
        j = 0
        coordi_j = list()
        while True:
            if i >= len(skeleton_idx[j]) - 1:
                coordi.append(coordi_j)
                coordi_j = []
                j += 1
                i = 0

            if j >= len(skeleton_idx):
                break

            check = skeleton_idx[j][i] 
            k = 0
            idx = 0
            while k < len(get_keypoints):
                if idx == len(get_keypoints[k]):
                    k+=1
                    idx = 0
                if check == -1:
                    coordi_j.append((0.0,0.0))
                    break
                elif check == get_keypoints[k][idx][-1] and k < 18:
                    coordi_j.append(get_keypoints[k][idx][0:2])
                    break

                idx += 1
            i += 1
        return coordi


    def recognizer(skeleton):
        
        #load models to action recognizer 
        skeleton = pose_normalization(skeleton)
        file_path_model = "models/action-openpose.hdf5"
        classes = ['punch', 'stand', 'wave']
        loadModels = load_model(file_path_model)
        Action_idx = np.argmax(loadModels.predict(np.array(skeleton).reshape(1,26)))
        predicted_label = classes[Action_idx]

        return predicted_label

    
    def drawActionResult(img_display, skeleton, str_action_type):
        #draw bounding box and display action result 
        font = cv2.FONT_HERSHEY_TRIPLEX 
        skeleton = list()
        for i in range(len(skeleton)):
            skeleton.append(skeleton[i][0])
            skeleton.append(skeleton[i][1])

        minx = 2000
        miny = 2000
        maxx = -2000
        maxy = -2000
        i = 0
        NaN = 0

        while i < len(skeleton):
            if not(skeleton[i][0]==NaN or skeleton[i][1]==NaN):
                minx = min(minx, skeleton[i][0])
                maxx = max(maxx, skeleton[i][0])
                miny = min(miny, skeleton[i][1])
                maxy = max(maxy, skeleton[i][1])
            i+=1


        minx = int(minx * img_display.shape[1])
        miny = int(miny * img_display.shape[0])
        maxx = int(maxx * img_display.shape[1])
        maxy = int(maxy * img_display.shape[0])
        print(minx, miny, maxx, maxy)
        
        # Draw bounding box
        # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
        img_display = cv2.rectangle(img_display,(int(minx), int(miny)),(int(maxx), int(maxy)),(255,0,0), 2)

        # Draw text at left corner


        """box_scale = max(0.5, min(2.0, (1.0*(maxx - minx)/img_display.shape[1] / (0.3))**(0.5) ))
        fontsize = 1 * box_scale
        linewidth = int(math.ceil(1 * box_scale))"""

        TEST_COL = int(200)
        TEST_ROW = int(200)

        img_display = cv2.putText(img_display, str_action_type, (TEST_COL, TEST_ROW), font, 3, (157, 50, 220), 1, cv2.LINE_AA)

    
    coordi = Skeleton(valid_pairs, invalid_pairs, keypoints_list, get_keypoints)

    for i in range(len(coordi)):
        action_type = recognizer(coordi[i])
        drawActionResult(img, coordi[i], action_type)
        
    return img

