import cv2

import sys
from os import getcwd
sys.path.append(getcwd() + "/tf-pose-estimation")

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator

import subprocess

def play_mp3(path):
    return subprocess.Popen(['mpg123', '-q', path])

def hardcode_dances(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video stream/file")
        return

    e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), (720,480))
    while cap.isOpened():
        ret_val, image = cap.read()
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

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
                yield("nothing")
                break

            wrists_y.sort()
            shoulders_y.sort()

            if wrists_y[-1] < shoulders_y[0]:
                yield "nae-nae"
                break
            else:
                yield "gangnam"
                break

if __name__ == "__main__":
    playing = None
    last_n_matched = 0
    last_n = "nothing"
    for move in hardcode_dances(0):
        if move == last_n:
            last_n_matched += 1
        else:
            last_n = move
            last_n_matched = 0

        if last_n_matched > 5:
            if playing is not None:
                playing.kill()

            if last_n == "nothing":
                playing = None
            else:
                playing = play_mp3(move + ".mp3")

            last_n_matched = 0

