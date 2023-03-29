#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import gc
import json
import pathlib
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, start_time):
        super().__init__()
        self.validation_data = validation_data
        self.start_time = start_time

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_mccs = []
        self.times = []
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = [np.argmax(x) for x in self.model.predict(self.validation_data[0], verbose=0)]

        val_f1 = f1_score(self.validation_data[1], predictions, average="macro")
        val_mcc = matthews_corrcoef(self.validation_data[1], predictions)

        self.val_f1s.append(val_f1)
        self.val_mccs.append(val_mcc)
        self.times.append(time.time() - self.start_time)

        print("- val_f1: %f - val_mcc %f" %(val_f1, val_mcc))

        tf.keras.backend.clear_session()
        gc.collect()

    def get_metrics(self):
        return self.val_f1s, self.val_mccs, self.times

def create_MLP(compiler):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    if compiler.lower() == "s":
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_MLP(input, output, optimizer):
    start = time.time()

    input = pathlib.Path(input)

    output = pathlib.Path(output)

    output.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(input/"x_train.csv", delimiter=",", dtype=int)
    X_cv = np.loadtxt(input/"x_cv.csv", delimiter=",", dtype=int)
    X_test = np.loadtxt(input/"x_test.csv", delimiter=",", dtype=int)

    y_train = np.loadtxt(input/"y_train.csv", delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_cv = np.loadtxt(input/"y_cv.csv", delimiter=",", dtype=int)
    y_cv_cat = tf.keras.utils.to_categorical(y_cv)
    y_test = np.loadtxt(input/"y_test.csv", delimiter=",", dtype=int)

    model = create_MLP(optimizer)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv_cat)).batch(64)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)

    custom_metrics = CustomCallback((val_dataset, y_cv), start)
    history = model.fit(train_dataset, 
            epochs=200, verbose = 2, 
            callbacks=[custom_metrics])
    
    metrics = custom_metrics.get_metrics()

    history.history["f1"] = metrics[0]
    history.history["mcc"] = metrics[1]
    history.history["times"] = metrics[2]

    history = json.dumps(history.history)
    
    f = open(output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')

    predictions = model.predict(X_test)

    predictions = [np.argmax(x) for x in predictions]
    print(confusion_matrix(y_test, predictions))
    print(f1_score(y_test, predictions, average="macro"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Processed dataset folder', default='dataset/one_hot_encoding/')
    parser.add_argument('-o', type=str, help='Output folder', default='results/')
    parser.add_argument('-g', type=str, help='Objective used slow SGD (S), fast Adam (A)', default="s")
    parser.add_argument('-t', type=str, help='Training type (a)all, (m)MLP, l(LR), r(RF)', default="a")

    args = parser.parse_args()

    train_MLP(args.f, args.o, args.g)
