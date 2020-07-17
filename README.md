# Action-recognition-using-pose-estimation-2D-openpose

# Introduction
* Used pre-trained models of OpenPose
* The model used OpenCV to read pre-trained so it was very slow because Opencv does not support GPU and hard to run in real-time. So if you want to run in real-time, you should read model with the orginal caffe.
* other repo about my action recognition: [here](https://github.com/TheK2NumberOne/action-recognition-project)

# Training with own dataset:
* Prepare data(actions) by running main.py, remember to uncomment the code of data collecting, the data will be saved as a .txt
* Do the training with the #action-openpose.ipynb in #/action/action_training/

# steps to take:
* Use Opencv to get all coordinates of joints on the human body
* Use an E_score to calculate the correlation of the pairs of joints, calculate which
joint A's connection to Joint B's would be best (because there are more than one
person in frame), if E_score > E_score_threshold is 0.1 and it satisfies the 7/10
points set forth when connection between joint A and joint B was found.
* Because there are so many people in the frame, i wrote a simple script to assemble
the right set of joints for each person.
* get the joint's positions.
* Pre-processing data: To remove all joints of the head such as nose, ears..etc,
because these positions are not needed in action recognition and fill -1 into the
missing joint coordinates. Finally the data will be normalized to about 0-1.
* Put processed joints into an neural network for training then use the softmax
function to classify actions.

# Dependencies:
* python >= 3.5
* Opencv >= 3.4.1
* tensorflow & keras
* numpy
    
# Acknowledge:
* [Online-Realtime-Action-Recognition-based-on-OpenPose](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose)
  
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  
  
