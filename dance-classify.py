import tensorflow as tf
from tensorflow.keras import layers

dance_moves = ['dab', 'nay-nay', 'whip', 'shuffling', 'moonwalk']
dance_moves_to_labels = {j: i for i, j in enumerate(dance_moves)}

BATCH_SIZE=32

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
#TODO: what is input shape? What are reasonable numbers?
layers.Dense(64, activation='relu', input_shape=(BATCH_SIZE,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with n output units:
layers.Dense(len(dance_moves), activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def feed_model(data, labels=None, epochs=None):
    if labels is None:
        #predict dance move
        result = model.predict(data, batch_size=BATCH_SIZE)
        return dance_moves[max(range(len(dance_moves)), key=lambda x: result[x])]
    else:
        lbls = [dance_moves_to_labels[i] for i in labels]
        if epochs is None: epochs = 1
        return model.fit(data, lbls, epochs=epochs)
