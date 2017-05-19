import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

from data import createTrainingDataset, TRAIN_DATABASE_PATH

PERCENT_TRAIN = .8
NUM_EPOCHS = 150
MODEL_BACKUP_FILE_NAME = 'backup.hdf'

def build12net():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(12,12,3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])
    
    return model

def train_classifier(saveFileName, scale, numEpochs, X_train, y_train, X_test, y_test, callbacks = None):
    getModel = {12: build12net}

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test,2)
    model = getModel.get(scale)()

    model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks, batch_size=32, epochs=numEpochs, verbose=1)
    return (hist, model.evaluate(X_test, y_test, verbose=0))

    
def train(saveFileName, scale, numEpochs = NUM_EPOCHS, verbose = True):
    if not os.path.exists(TRAIN_DATABASE_PATH):
        createTrainingDataset(scale)

    with h5py.File(TRAIN_DATABASE_PATH, 'r') as infile:
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
            ModelCheckpoint(saveFileName, monitor='val_acc', save_best_only=True, verbose=1),
        ]

        hist, score = train_classifier(saveFileName, scale, numEpochs, X_train, y_train, X_test, y_test, callbacks)

        if (verbose):
            plt.plot(hist.history['val_loss'])
            plt.show()
            print('Test set accuracy:', score[1])