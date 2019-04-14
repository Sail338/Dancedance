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
    else:
        print("opened stream/file")

    e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), (720,480))
    while cap.isOpened():
        ret_val, image = cap.read()
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # print(humans)
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

            if wrists_y[-1] > shoulders_y[0]:
                yield "gangnam"
                break
            else:
                yield "nae-nae"
                break

def play_manager(webcam_interpreter):
    playing = None
    prevPlayThis = None
    last_n = "nothing"
    window = []
    refreshCtr = 0
    for move in webcam_interpreter(0):
        print("a move")

        #make sure only 10 things in "window"
        if len(window) <10:
            window.append(move)
        else:
            window.append(move)
            window.pop(0)

        #get the mode. we need this b/c regular mode
        #function craps out if multiple modes
        gangCtr = 0
        naeCtr = 0
        nothingCtr = 0
        for x in window:
            if x == "gangnam":
                gangCtr+=1
            elif x == "nae-nae":
                naeCtr+=1
            elif x == "nothing":
                nothingCtr+=1
            else:
                print("this shouldn't happen!")

        #get the maximum of the 3 values
        #choose the dance move if the same number as nothing
        playThis = "nothing"
        maxMove = max([gangCtr, naeCtr, nothingCtr])
        if maxMove == gangCtr:
            playThis = "gangnam"
        elif maxMove == naeCtr:
            playThis = "nae-nae"
        else:
            playThis = "nothing"

        refreshCtr+=1
        if refreshCtr > 20:
            print("refreshing song (or not)")
            #if playing is not None:
            #    playing.kill()
            refreshCtr = 0
            if playThis != prevPlayThis and playing is not None:
                playing.kill()

            if playThis == "nothing":
                playing = None
            else:
                playing = play_mp3(playThis + ".mp3")
                print("playing ", playThis)

            prevPlayThis = playThis

if __name__ == "__main__":
    play_manager(hardcode_dances)
