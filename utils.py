import cv2 
import os 
import sys

cam_width = 1280
cam_height = 720

def choose_run_mode(args):
    if args.video:
        if not os.path.isfile(args.video):
            print("input video file does not exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
    
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    return cap

def load_pretraind_openpose():
    try:
        protoFile = "pose/graph_models/pose_deploy_linevec.prototxt"
        weightsFile = "pose/graph_models/pose_iter_440000.caffemodel"
    except:
        print("can't open file. Check again")
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    return net


