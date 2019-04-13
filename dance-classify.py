import cv2

import sys
from os import getcwd
sys.path.append(getcwd() + "/tf-pose-estimation")

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose.common import CocoPart

import tensorflow as tf
from tensorflow.keras import layers

import estimateBoundingBox as ebb

dance_moves = ['dab', 'nae nae', 'whip', 'shuffling', 'moonwalk','moonwalk', 'sprinkler','macarena','twerking','flossing','gangnam style']
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

class Person:
    CONFIDENCE_THRES = 0.3
    def __init__(self, human):
        if CocoPart.Neck not in human.body_parts or human.body_parts[CocoPart.Neck].score < Person.CONFIDENCE_THRES:
            self.ok = False #no neck, no life
            return
        self.ok = True
        self.neck = human.body_parts[CocoPart.Neck]
        self.bb = None
        self.frame = []
        self.epochs = 0
    def set_bb_from(self, person):
        if person.bb is None:
            return
        delta_x = self.neck.x - person.neck.x
        delta_y = self.neck.y - person.neck.y
        self.bb = person.bb.copy()
        self.bb['x'] += delta_x
        self.bb['y'] += delta_y
    def add_frame(self, human):
        self.epochs += 1
        new_frames = [([p.x, p.y] if p.score > Person.CONFIDENCE_THRES else [-1, -1]) for p in human.pairs]
        for p in new_frames:
            self.frames.append((p[0] - self.bb['x'])/self.bb['w'])
            self.frames.append((p[1] - self.bb['y'])/self.bb['h'])
    def dist(self, person):
        return ebb.distanceFormula(self.neck.x, self.neck.y, person.neck.x, person.neck.y)
    def __eq__(self, other):
        return self.dist(other) == 0

def read_video(video_file, model, target_size):
    HEART_DIST_TOL = 5
    CONFIDENCE = 0.3 #IDK, they use this somewhere
    cap = cv2.VideoCapture(video_file)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    num_frames = 0
    persons = []
    while cap.isOpened():
        new_peeps = []
        ret_val, image = cap.read()
        e = TfPoseEstimator(get_graph_path(model), target_size=target_size)

        humans = e.inference(image,resize_to_default=True, upsample_size=4.0)
        print(humans)
        for human in humans:
            curr_person = Person(human)
            if not curr_person.ok:
                continue

            guess_person = min(range(len(persons)), key=lambda i: person[i].dist(curr_person))
            if persons[guess_person].dist(curr_person) > HEART_DIST_TOL:
                #new person
                new_bb = ebb.estimateBoundingBox(human)
                if new_bb is None:
                    continue
                else:
                    curr_person.bb = new_bb
            else:
                curr_person.set_bb_from(persons[guess_person])
            new_peeps.append(curr_person)
            curr_person.add_frame(human)

            for peep in persons:
                if peep not in new_peeps:
                    yield peep.frames, peep.epochs

def save_model(file="./moderu"):
    model.save_weights(file)

def restore_model(file="./moderu"):
    model.load_weights(file)

if __name__ == "__main__":
    from search_downloader import search_n_dl
    from os import listdir, mkdir
    from os.path import isfile, join, exists

    if exists("moderu"):
        restore_model("moderu/ore")
    else:
        mkdir("moderu")
        save_model("moderu/ore")

    for move in dance_moves:
        try:
            mkdir("moves/"+move)
            search_n_dl(move + " compilation", 20, "moves/"+move)
        except FileExistsError:
            print("Directory Already Exists")
        continue
        for vid in listdir(move):
            if isfile(join(move, vid)):
                all_the_data = read_video(join(move, vid), 'cmu', (720, 480))
                for datum, epochs in all_the_data:
                    feed_model(datum, move, epochs)
    save_model("moderu/ore")
