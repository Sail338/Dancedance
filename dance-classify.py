import cv2
#TODO: fix import?

import sys
from os import getcwd
sys.path.append(getcwd() + "/tf-pose-estimation")

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator

import tensorflow as tf
from tensorflow.keras import layers

dance_moves = ['dab', 'nay-nay', 'whip', 'shuffling', 'moonwalk', 'what the good fuck']
dance_moves_to_labels = {j: i for i, j in enumerate(dance_moves)}

BATCH_SIZE=34

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(BATCH_SIZE,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with n output units:
layers.Dense(len(dance_moves), activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def zero_except(idx):
    rv = [0 for i in dance_moves]
    rv[idx] = 1
    return rv

def feed_model(data, labels=None, epochs=None):
    if labels is None:
        #predict dance move
        result = model.predict(data, batch_size=BATCH_SIZE)
        return dance_moves[max(range(len(dance_moves)), key=lambda x: result[x])]
    else:
        lbls = [zero_except(dance_moves_to_labels[i]) for i in labels]
        if num_frames is None: num_frames = 1
        return model.fit(data, lbls, epochs=num_frames)

def read_video(video_file, model, target_size):
    CONFIDENCE = 0.4
    cap = cv2.VideoCapture(video_file)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    frames = []
    num_frames = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        e = TfPoseEstimator(get_graph_path(model), target_size=target_size)

        humans = e.inference(image)
        temp_frames = []
        frame_is_trash = True
        for part in humans:
            if part.score > CONFIDENCE:
                frame_is_trash = False
                temp_frames += [part.x, part.y]
            else:
                temp_frames += [-1, -1]
        if not frame_is_trash:
            num_frames += 1
            frames += temp_frames

def save_model(file="moderu"):
    model.save_weights(file)

def restore_model(file="moderu"):
    model.load_weights(file)

if __name__ == "__main__":
    from search_downloader import search_n_dl
    from os import listdir, mkdir
    from os.path import isfile, join, exists

    if exists("moderu"):
        restore_model()
    move = "nae-nae"
    if not exists(move):
        mkdir(move)
        size = (480, 480)#ANDREA PLZ
        search_n_dl(move, 10, move)

    for vid in listdir(move):
        if isfile(join(move, vid)):
            data, num = read_video(join(move, vid), 'cmu', size)
            feed_model(data, move, num)
    save_model()
