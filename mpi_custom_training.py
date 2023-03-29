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
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

tf.keras.utils.set_random_seed(7)

def create_MLP():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    return model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('-e', type=int, help='Epochs number', default=100000)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-b', type=int, help='Batch size', default=64)

args = parser.parse_args()

epochs = args.e
learning_rate = args.l
dataset = args.d
output = args.o
batch_size = args.b

output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
dataset = pathlib.Path(dataset)

model = create_MLP()
start = time.time()

if rank == 0:
    node_weights = []
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    for node in range(1, size):
        node_weights.append(comm.recv(source=node, tag=1000))
    
    total_size = sum(node_weights)

    node_weights = [weight/total_size for weight in node_weights]
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
    results["times"]["loads"].append(time.time() - start)

else:
    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    comm.send(len(X_train), dest=0, tag=1000)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

model.set_weights(comm.bcast(model.get_weights(), root=0))
if rank == 0:
    results["times"]["loads"].append(time.time() - start)

batch = 0
for epoch in range(epochs):
    weights = None
    if rank == 0:
        for node in range(1, size):
            grads = comm.recv(source=node, tag=epoch)

            if node == 1:
                avg_grads = grads
            else:
                avg_grads = [ avg_grads[i] + grads[i] for i in range(len(grads))]
        
        avg_grads = [grads/(size-1) for grads in avg_grads]

        optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))
        weights = model.get_weights()
        
    else:
        x_batch_train, y_batch_train = train_dataset[batch]

        with tf.GradientTape() as tape:

            logits = model(x_batch_train, training=True) 
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)

        comm.send(grads, dest=0, tag=epoch)
        batch = (batch + 1) % len(train_dataset)


    weights = comm.bcast(weights, root=0)

    model.set_weights(weights)

    if rank == 0 and epoch % 1500 == 0:
        print("\n End of epoch %d" % epoch)
        predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
        train_f1 = f1_score(y_cv, predictions, average="macro")
        train_mcc = matthews_corrcoef(y_cv, predictions)
        train_acc = accuracy_score(y_cv, predictions)

        results["acc"].append(train_acc)
        results["f1"].append(train_f1)
        results["mcc"].append(train_mcc)
        results["times"]["epochs"].append(time.time() - start)
        print("- val_f1: %f - val_mcc %f - val_acc %f" %(train_f1, train_mcc, train_acc))

    
    tf.keras.backend.clear_session()
    gc.collect()

if rank==0:
    history = json.dumps(results)

    f = open( output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')