#!/usr/bin/env python3.5
import numpy as np
import cv2
import h5py
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf

from sklearn.model_selection import train_test_split

from data import SCALES, FACE_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS, CALIBRATION_DATABASE_PATHS, DATASET_LABEL, LABELS_LABEL, RANDOM_SEED, BATCH_SIZE
from FaceDetection import TRAIN, DEBUG

PERCENT_TRAIN = .8
NUM_EPOCHS = 300
OPTIMIZER = SGD()
MAIN_INPUT_LAYER_NAME = 'input0'
SECONDARY_INPUT_LAYER_NAME = 'input1'
INPUT_LAYER_NAME_ID = 'input'

def build12net():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12, 12, 3), name=MAIN_INPUT_LAYER_NAME))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

def build12calibNet():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12, 12, 3), name=MAIN_INPUT_LAYER_NAME))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(45, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

def build24net():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(24,24,3), name=MAIN_INPUT_LAYER_NAME))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    secondaryInput = Input(shape=(12,12,3), name=SECONDARY_INPUT_LAYER_NAME)
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
    finalModel.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return finalModel

def build24calibNet():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(24,24,3), name=MAIN_INPUT_LAYER_NAME))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(45, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

models = {False: {SCALES[0][0]: build12net, SCALES[1][0]: build24net}, 
          True: {SCALES[0][0]: build12calibNet, SCALES[1][0]: build24calibNet}}

def preprocessImages(X):
    return X/255

def getTrainingSets(stageIdx, trainCalib, *args):
    if not trainCalib:
        (faceDatasetFileName, negDatasetFileName) = args
        with h5py.File(faceDatasetFileName, 'r') as faces, h5py.File(negDatasetFileName, 'r') as negatives:
            faceDataset, negDataset = (faces[DATASET_LABEL], negatives[DATASET_LABEL])
            X = preprocessImages(np.vstack((faceDataset, negDataset)))
            y = np.vstack((np.ones((len(faceDataset), 1)), np.zeros((len(negDataset), 1))))
    else:
        calibDatasetFileName = args[0]
        with h5py.File(calibDatasetFileName, 'r') as calibSamples:
            X = preprocessImages(calibSamples[DATASET_LABEL][:])
            y = calibSamples[LABELS_LABEL][:]
    
    X_secondary = np.zeros((len(X), SCALES[0][0], SCALES[0][1], 3))
    useSecondaryInput = stageIdx >= 1 and not trainCalib

    if (useSecondaryInput):
        resizeTo = SCALES[stageIdx-1]
        X_secondary = np.zeros((len(X), resizeTo[1], resizeTo[0], 3))
        for i in np.arange(len(X_secondary)):
            X_secondary[i] = cv2.resize(X[i], resizeTo)

    shuffledDataset = train_test_split(X, X_secondary, y, train_size = PERCENT_TRAIN, random_state = RANDOM_SEED)
    X_train, X_test, X_secondary_train, X_secondary_test, y_train, y_test = shuffledDataset

    if (useSecondaryInput):
        X_train = [X_train, X_secondary_train]
        X_test = [X_test, X_secondary_test]

    return X_train, X_test, y_train, y_test 

def trainModel(model, X_train, X_test, y_train, y_test, numEpochs, callbacks = None):
    numCategories = int(np.amax(y_test))+1
    y_train = np_utils.to_categorical(y_train, numCategories)
    y_test = np_utils.to_categorical(y_test, numCategories)

    model.fit(X_train, y_train, validation_data = (X_test, y_test), callbacks = callbacks, batch_size = BATCH_SIZE, epochs = numEpochs, verbose = 2)


def train(stageIdx, trainCalib, numEpochs = NUM_EPOCHS, debug = DEBUG):
    from detect import NET_FILE_NAMES

    scale = SCALES[stageIdx][0]
    fileName = NET_FILE_NAMES[trainCalib][scale]
    faceDatasetFileName = FACE_DATABASE_PATHS[stageIdx]
    negDatasetFileName = NEGATIVE_DATABASE_PATHS[stageIdx]
    calibDatasetFileName = CALIBRATION_DATABASE_PATHS.get(scale)

    fileNames = (faceDatasetFileName, negDatasetFileName) if not trainCalib else (calibDatasetFileName,)

    X_train, X_test, y_train, y_test = getTrainingSets(stageIdx, trainCalib, *fileNames)
    callbacks = [ModelCheckpoint(fileName if not DEBUG else 'debug.hdf', monitor='val_loss', save_best_only=True, verbose=1)]
    model = models[trainCalib].get(scale)()

    trainModel(model, X_train, X_test, y_train, y_test, numEpochs, callbacks)


