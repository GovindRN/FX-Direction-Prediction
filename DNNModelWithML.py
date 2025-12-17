import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models 
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

def set_seeds(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def cw(df):
    c0, c1 = np.bincount(df["dir"])
    w0 = (1/c0) * (len(df)) / 2
    w1 = (1/c1) * (len(df)) / 2
    return {0:w0, 1:w1}

def create_model(hl=10, hu=100, dropout=False, rate=0.3, regularize=False, reg=regularizers.l1(0.0005), input_dim=None):
    optimizer = optimizers.Adam(learning_rate=0.0001)
    if not regularize:
        reg = None
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hu, activation="relu", activity_regularizer=reg)(inputs)
    if dropout:
        x = layers.Dropout(rate, seed=100)(x)
    for _ in range(hl):
        x = layers.Dense(hu, activation="relu", activity_regularizer=reg)(x)
        if dropout:
            x = layers.Dropout(rate, seed=100)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def create_sequential_model(hl=10, hu=100, dropout=False, rate=0.3, regularize=False, reg=regularizers.l1(0.0005), input_dim=None):
    optimizer = optimizers.Adam(learning_rate=0.0001)
    if not regularize:
        reg = None
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(hu, activation="relu", activity_regularizer=reg))
    if dropout:
        model.add(layers.Dropout(rate, seed=100))
    for _ in range(hl):
        model.add(layers.Dense(hu, activation="relu", activity_regularizer=reg))
        if dropout:
            model.add(layers.Dropout(rate, seed=100))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model