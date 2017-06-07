#!/usr/bin/env python3.5
import numpy as np
import cv2
import h5py
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import tensorflow as tf

from sklearn.model_selection import train_test_split

from data import SCALES, FACE_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS, DATASET_LABEL, RANDOM_SEED, BATCH_SIZE
from FaceDetection import TRAIN

PERCENT_TRAIN = .8
NUM_EPOCHS = 300

def build12net():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12, 12, 3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def build12calibNet():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12, 12, 3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(45, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def build24net():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(24,24,3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    secondaryInput = Input(shape=(12,12,3))
    convLayer = Conv2D(16, (3, 3), activation='relu')(secondaryInput)
    poolingLayer = MaxPooling2D((3,3), strides=2)(convLayer)

    flattened = Flatten()(poolingLayer)
    secondaryOutput = Dense(16, activation='relu')(flattened)
    dropoutLayer = Dropout(0.3)(secondaryOutput)

    primaryInput = Input(shape=(24,24,3))
    merged = concatenate([model(primaryInput), dropoutLayer])
    finalDropout = Dropout(0.3)(merged)
    output = Dense(2, activation='softmax')(finalDropout)

    finalModel = Model(inputs=[primaryInput, secondaryInput], outputs=output)
    finalModel.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return finalModel

def build24calibNet():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(24,24,3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(45, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

models = {SCALES[0][0]: build12net}

def getTrainingSets(faceDatasetFileName, negDatasetFileName):
    with h5py.File(faceDatasetFileName, 'r') as faces, h5py.File(negDatasetFileName, 'r') as negatives:
        faceDataset, negDataset = (faces[DATASET_LABEL], negatives[DATASET_LABEL])
        X = np.vstack((faceDataset, negDataset))
        y = np.vstack((np.ones((len(faceDataset), 1)), np.zeros((len(negDataset), 1))))

        sess = tf.Session()
        with sess.as_default():
            X = tf.map_fn(lambda img: tf.image.per_image_standardization(img), X.astype(np.float32)).eval()

        return train_test_split(X, y, train_size = PERCENT_TRAIN, random_state = RANDOM_SEED)

def trainModel(model, X_train, X_test, y_train, y_test, numEpochs, callbacks = None):
    numCategories = int(np.amax(y_test))+1
    y_train = np_utils.to_categorical(y_train, numCategories)
    y_test = np_utils.to_categorical(y_test, numCategories)

    model.fit(X_train, y_train, validation_data = (X_test, y_test), callbacks = callbacks, batch_size = BATCH_SIZE, epochs = numEpochs, verbose = 2)


def train(stageIdx, numEpochs = NUM_EPOCHS):
    from detect import NET_FILE_NAMES

    scale = SCALES[stageIdx][0]
    fileName = NET_FILE_NAMES[scale]
    faceDatasetFileName = FACE_DATABASE_PATHS[stageIdx]
    negDatasetFileName = NEGATIVE_DATABASE_PATHS[stageIdx]

    X_train, X_test, y_train, y_test = getTrainingSets(faceDatasetFileName, negDatasetFileName)
    callbacks = [ModelCheckpoint(fileName, monitor='val_loss', save_best_only=True, verbose=1)]
    model = models.get(scale)()

    trainModel(model, X_train, X_test, y_train, y_test, numEpochs, callbacks)


