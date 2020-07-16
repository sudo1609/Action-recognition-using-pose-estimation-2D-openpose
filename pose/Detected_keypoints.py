import cv2
import numpy as np 
from pose.format_coco import Points

def getKeyPoints(probMap, threshold = 0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold) 
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    keypoints = list()
    for contour in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, contour, 1)
        maskedProbMap = mapSmooth * blobMask
        _, _ , _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints


def detect_keypoints(image, output):
    detected_keypoints = list()
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1
    keypoints = []
    for part in range(Points):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image.shape[1], image.shape[0]))
    #   plt.figure()
    #   plt.imshow(255*np.uint8(probMap>threshold))
        keypoints = getKeyPoints(probMap, threshold)
        #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
        detected_keypoints.append(keypoints_with_id)
    return detected_keypoints, keypoints_list