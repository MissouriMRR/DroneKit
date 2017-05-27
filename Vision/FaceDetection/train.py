import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import Callback

from data import createTrainingDataset, createCalibrationDataset, getFaceAnnotations, TRAIN_DATABASE_PATH, CALIBRATION_DATABASE_PATHS

PERCENT_TRAIN = .8
NUM_EPOCHS = 300

def build12net():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12,12,3)))
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
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12,12,3)))
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

def trainModel(saveFileName, scale, numEpochs, X_train, y_train, X_test, y_test, trainCalib, callbacks = None):
    if scale >= 24 and not trainCalib:
        newSize = {24:12}.get(scale)
        vecs = (X_train, X_test)
        resized = [np.ones((vec.shape[0], newSize, newSize, 3), vec.dtype) for vec in vecs]

        for i, vec in enumerate(vecs):
            for j in np.arange(vec.shape[0]):
                resized[i][j] = cv2.resize(vec[j], (newSize,newSize))
        
        X_train = [X_train, resized[0]]
        X_test = [X_test, resized[1]]
        
    getModel = {12: build12net, 24: build24net} if not trainCalib else {12: build12calibNet, 24: build24calibNet}
    numCategories = 2 if not trainCalib else 45

    y_train = np_utils.to_categorical(y_train, numCategories)
    y_test = np_utils.to_categorical(y_test, numCategories)
    model = getModel.get(scale)()

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks, batch_size=32, epochs=numEpochs, verbose=1)
    return (hist, model.evaluate(X_test, y_test, verbose=0))

def train(saveFileName, scale, verbose = True, trainCalib = False, numEpochs = NUM_EPOCHS):
    dbPath = TRAIN_DATABASE_PATH if not trainCalib else CALIBRATION_DATABASE_PATHS.get(scale)

    if not os.path.exists(dbPath):
        if not trainCalib:
            createTrainingDataset(scale)
        else:
            createCalibrationDataset(getFaceAnnotations(), scale)

    with h5py.File(dbPath, 'r') as infile:
        db = infile['data']
        y = infile['labels']
        numTrainImages = int(len(db)*PERCENT_TRAIN)
        X_train = db[:numTrainImages]
        y_train = y[:numTrainImages]
        X_test = db[numTrainImages:]
        y_test = y[numTrainImages:]

        X_train = X_train/255
        X_test = X_test/255

        callbacks = [
            ModelCheckpoint(saveFileName, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        ]

        hist, score = trainModel(saveFileName, scale, numEpochs, X_train, y_train, X_test, y_test, trainCalib, callbacks)

        if (verbose):
            plt.plot(hist.history['val_loss'])
            plt.show()
            print('Test set accuracy:', score[1])