import os
import sys
import time
import pickle
import tensorflow as tf
import pickle
import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential

sys.setrecursionlimit(1500)



class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)




def data_generator():
    with open('preprocessed_dataset_train.h5', 'rb') as data:
        dataset = pickle.load(data)
    
    with open('preprocessed_dataset_valid.h5', 'rb') as data_valid:
        dataset_valid = pickle.load(data_valid)

    classes = np.array(list(set(set(dataset['Y']))))

    enc = LabelEncoder()
    enc.fit(dataset['Y'])
    dataset['Y'] = enc.transform(dataset['Y'])
    ohe = OneHotEncoder().fit(dataset['Y'].reshape(-1, 1))
    dataset['Y'] = ohe.transform(dataset['Y'].reshape(-1, 1)).toarray()

    dataset_valid['Y'] = enc.transform(dataset_valid['Y'])
    dataset_valid['Y'] = ohe.transform(dataset_valid['Y'].reshape(-1, 1)).toarray()

    print(dataset['X'].shape, dataset['Y'].shape)
    print(dataset_valid['X'].shape, dataset_valid['Y'].shape)

    print(classes)
    return dataset, dataset_valid, classes

def build_model(input_shape, classes):

    model = Sequential()

    model.add(keras.layers.LSTM(64, input_shape = input_shape, return_sequences = True))
    model.add(keras.layers.LSTM(64))

    
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(classes, activation='sigmoid'))

    optimiser = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimiser, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model


def train_model(epochs = 10):
    time_callback = TimeHistory()

    train_generator, validation_generator, classes = data_generator()
    model = build_model(input_shape = (21, 3), classes = len(classes))

    r = model.fit(
        x=train_generator['X'],
        y=train_generator['Y'],
        epochs = epochs,
        batch_size=64,
        shuffle=True,
        validation_data=(validation_generator['X'], validation_generator['Y']),
        validation_batch_size = 128,
        callbacks=[time_callback]
    )

    r.history['time'] = time_callback.times

    model.save('sequencial_model.h5')
    pickle.dump(list(r.history.items()),open('sequencial_model_score.h5', 'wb'))


if __name__ == "__main__":
    train_model(60)
