import cv2

import pickle
import numpy as np

import sys
import threading
from os import getcwd
import os
from os.path import exists
sys.path.append(getcwd() + "/tf-pose-estimation")

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator, BodyPart
from tf_pose.common import CocoPart

import tensorflow as tf
from tensorflow.keras import layers

import estimateBoundingBox as ebb

#dance_moves = ["nae nae", 'shuffling dance', 'moonwalk', 'sprinkler dance','macarena','twerking','flossing','gangnam style']
dance_moves = ["naenae",'gangam']
dance_moves_to_labels = {j: i for i, j in enumerate(dance_moves)}

BATCH_SIZE = 38

TICKET = 1
inference_res = []
if os.path.isfile('inference_res.pkl'):
    print("YES YES YES YES YES YES")
    try:
        with open("./inference_res.pkl",'rb') as inf:
            inference_res = pickle.load(inf)
            print(len(inference_res))
    except Exception as e:
        print(str(e))
        pass
lock = threading.Lock()

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

def feed_model(data, labels=None, epochs=None, steps_per_epoch=None):
    data2 = np.array([data])
    if labels is None:
        #predict dance move
        result = model.predict(data2)
        return dance_moves[max(range(len(dance_moves)), key=lambda x: result[0][x])]
    else:
        lbls = np.array([zero_except(dance_moves_to_labels[i]) for i in labels])
        print(lbls, epochs, data2.shape)
        if epochs is None: epochs = 1
        return model.fit(data2, lbls, epochs=epochs, steps_per_epoch=steps_per_epoch)

class Person:
    CONFIDENCE_THRES = 0.0
    def __init__(self, human):
        self.ok = False
        self.neck = BodyPart(0, CocoPart.Neck, 0, 0, 1)
        self.bb = None
        self.frames = []
        self.epochs = 0
        parts = [human.body_parts[part] for part in human.body_parts]
        for part in parts:
            part_name = str(part.get_part_name())
            if "Neck" in part_name:
                self.ok = True
                self.neck = part
                break
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
        all_parts = {str(p): i for i, p in enumerate(CocoPart)}
        new_frames = [(-1, -1) for i in CocoPart]
        for part in human.body_parts:
            pp = human.body_parts[part]
            new_frames[all_parts[str(pp.get_part_name())]] = (pp.x, pp.y)

        for p in new_frames:
            if p[0] >= 0:
                self.frames.append((p[0] - self.bb['x'])/self.bb['w'])
                self.frames.append((p[1] - self.bb['y'])/self.bb['h'])
            else:
                self.frames.append(-1)
                self.frames.append(-1)

    def dist(self, person):
        return ebb.distanceFormula(self.neck.x, self.neck.y, person.neck.x, person.neck.y)
    def __eq__(self, other):
        return self.dist(other) == 0

def inference(order,image,model,target_size,est):
    global TICKET
    humans = est.inference(image,resize_to_default=True, upsample_size=4.0)
    while order != TICKET:
        continue
    with lock:
        TICKET+=1
        inference_res.append(humans)
        if order % 5000 == 0:
            print("Frame "+str(order)+" Finished...")


    return

def read_video(video_file, model, target_size):
    global inference_res 
    HEART_DIST_TOL = 100
    CONFIDENCE = 0.3 #IDK, they use this somewhere
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, len(inference_res))

    if cap.isOpened() is False:
        print("Error opening video streasm or file")

    persons = []
    frame_num = TICKET
    threads = []
    est = TfPoseEstimator(get_graph_path(model), target_size=target_size)
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        thread = threading.Thread(target=inference, args=(frame_num,image,model,target_size,est,))
        frame_num+=1
        threads.append(thread)
        #humans = e.inference(image,resize_to_default=True, upsample_size=4.0)
        thread.start()
        if len(threads) % 15000 == 0:
            for thread in threads:
                thread.join()
            threads = []
    for thread in threads:
        thread.join()
    print("All Threads completed")
    for humans in inference_res:
        unpaired = set(range(len(persons)))
        print(unpaired)
        new_peeps = []
        for human in humans:
            curr_person = Person(human)
            if not curr_person.ok:
                print("neckless fucktard")
                continue

            if persons:
                guess_person = min(range(len(persons)), key=lambda i: persons[i].dist(curr_person))
            if (len(persons) == 0) or persons[guess_person].dist(curr_person) > HEART_DIST_TOL:
                #new person
                new_bb = ebb.getUserBoundingBox(human)
                if new_bb is None:
                    continue
                else:
                    curr_person.bb = new_bb
            else:
                curr_person.set_bb_from(persons[guess_person])
                if guess_person in unpaired: unpaired.remove(guess_person)
            new_peeps.append(curr_person)
            curr_person.add_frame(human)
            
        for idx in unpaired:
            yield persons[idx].frames, persons[idx].epochs
        persons = new_peeps

    for peep in persons:
        print("A person yote")
        yield peep.frames, peep.epochs
    inference_res = []


def save_model(file="./moderu"):
    model.save_weights(file)

def restore_model(file="./moderu"):
    model.load_weights(file)

def webcam():
    model_used = "mobilenet_v2_large"
    e = TfPoseEstimator(get_graph_path(model_used), target_size=(720, 480))
    cam = cv2.VideoCapture(0)
    pers = None
    while True:
        ret_val, image = cam.read()
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        with_bb = filter(lambda hb: hb[1] is not None,
                ((h, ebb.getUserBoundingBox(h)) for h in humans))
        try:
            middle_man, bb = min(with_bb, key=lambda wb: ebb.distanceFormula(wb[1]['x'], wb[1]['y'], 0.5, 0.5))
        except ValueError:
            yield "nothing"
        else:
            if pers is None:
                pers = Person(middle_man)
            pers.bb = bb
            pers.add_frame(middle_man)
            print(len(pers.frames))
            if len(pers.frames) >= BATCH_SIZE:
                yield feed_model(pers.frames.copy())
                pers = Person(middle_man)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from derp import play_manager
        restore_model("moderu/ore")
        play_manager(lambda v: map(lambda s: s.replace(" ", "-"), webcam()))

    # from search_downloader import search_n_dl
    from os import listdir, mkdir
    from os.path import isfile, join, exists

    if exists("moderu"):
        restore_model("moderu/ore")
    else:
        mkdir("moderu")
        save_model("moderu/ore")

    try:
        for move in dance_moves:
            try:
                mkdir("moves/"+move)
                search_n_dl(move + " compilation", 20, "moves/"+move)
            except FileExistsError:
                print("Directory Already Exists")
            move_path = "moves/" + move
            for vid in listdir(move_path):
                print(vid)
                if isfile(join(move_path, vid)):
                    all_the_data = read_video(join(move_path, vid), 'mobilenet_thin', (368, 368))
                    for datum, epochs in all_the_data:
                        if len(datum) >= BATCH_SIZE:
                            feed_model(datum, labels=[move for i in range(epochs)], steps_per_epoch=epochs)
                    save_model("moderu/ore")
                    print("FINISHED A VIDEO")
    except Exception as e:
        import traceback as tb
        print(tb.format_exc())
        print("SHIT HAPPENED")
        save_model("moderu/ore")
        with open("inference_res.pkl", "wb") as inf:
            pickle.dump(inference_res, inf)
