import cv2

import sys
from os import getcwd
sys.path.append(getcwd() + "/tf-pose-estimation")

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator

import tensorflow as tf
from tensorflow.keras import layers



def hardcode_dances(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video stream/file")
        return


    e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), (720,480))
    while cap.isOpened():
        ret_val, image = cap.read()
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)

        for human in humans:
            parts_dict = human.body_parts
            wrists_y = []
            shoulders_y = []
            for k,v in parts_dict.items():
                if 'Wrist' in str(v.get_part_name()) or 'Elbow' in str(v.get_part_name()):
                    wrists_y.append(v.y)
                elif 'Shoulder' in str(v.get_part_name()) or 'Neck' in str(v.get_part_name()):
                    shoulders_y.append(v.y)

            #if all wrists are higher up than all shoulders, print "xD"
            #else, print ":c"

            if len(wrists_y) == 0 or len(shoulders_y) == 0:
                print("nothing")
                continue

            wrists_y.sort()
            shoulders_y.sort()

            if wrists_y[-1] < shoulders_y[0]:
                print("nae nae")
            else:
                print("gangnam")

if __name__ == "__main__":
    hardcode_dances(0)
