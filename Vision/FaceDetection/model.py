import pickle
import os
import multiprocessing
import cv2
import numpy as np
import atexit
import copy
import time
from abc import ABC, abstractmethod

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
atexit.register(K.clear_session)

from hyperopt_keras import HyperoptWrapper, getBestParams, parseParams, tune
from FaceDetection import DEBUG
from preprocess import ImageNormalizer
from data import SCALES, DATASET_LABEL
from util import TempH5pyFile
hp = HyperoptWrapper()

DROPOUT_PARAM_ID = 'dropout'
OPTIMIZER_PARAMS = ['lr', 'momentum', 'decay', 'nesterov']
NORMALIZATION_PARAMS = ['norm', 'flip']
TRAIN_PARAMS = ['batchSize']

OPTIMIZER = SGD

DEFAULT_BATCH_SIZE = 128
PREDICTION_BATCH_SIZE = 4096
DEFAULT_NUM_EPOCHS = 300
DEFAULT_Q_SIZE = 1024
DEBUG_FILE_PATH = 'debug.hdf'
PARAM_FILE_NAME_FORMAT_STR = '%sparams'

STAGE_ONE_NOT_TRAINED_ERROR = 'You must train the stage one models before moving onto stage two!'

NUM_CLASSIFIER_CATEGORIES = 2
NUM_CALIBRATOR_CATEGORIES = 45

class ObjectClassifier(ABC):
    PARAM_SPACE = {
        'dropout0': hp.uniform(0, .5),
        'dropout1': hp.uniform(0, .5),
        'lr': hp.loguniform(1e-4, .1),
        'batchSize': hp.choice(32, 64, 128, 256),
        'norm':  hp.choice(ImageNormalizer.STANDARD_NORMALIZATION, ImageNormalizer.MIN_MAX_SCALING),
        'flip': hp.choice(None, ImageNormalizer.FLIP_HORIZONTAL)
    }

    DEFAULT_DROPOUT = .3
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

    def __init__(self, stageIdx):
        self.imgSize = SCALES[stageIdx]
        self.inputShape = (self.imgSize[1], self.imgSize[0], 3)
        self.trainedModel = None
        self.bestParams = None

        self.update()

    @abstractmethod
    def __call__(self):
        pass
    
    def getParamFilePath(self):
        return PARAM_FILE_NAME_FORMAT_STR % (os.path.splitext(self.getWeightsFilePath())[0],)

    def getParamSpace(self):
        return self.PARAM_SPACE

    def getWeightsFilePath(self):
        from detect import NET_FILE_NAMES
        return NET_FILE_NAMES[isinstance(self, ObjectCalibrator)][SCALES[self.stageIdx][0]]

    def getNormalizationMethod(self):
        return self.bestParams['norm'] if self.wasTuned() else ImageNormalizer.STANDARD_NORMALIZATION

    def getNormalizationParams(self):
        params = {}

        if self.wasTuned():
            params = {k: v for k, v in self.bestParams.items() if k in NORMALIZATION_PARAMS}
            del params['norm']

        return params

    def getDropouts(self):
        dropouts = [self.DEFAULT_DROPOUT] * len([k for k in self.getParamSpace() if k.startswith(DROPOUT_PARAM_ID)])

        if self.wasTuned():
            dropouts = []
            for k, v in self.bestParams.items():
                if type(k) is str and k.startswith(DROPOUT_PARAM_ID):
                    idx = int(k.replace(DROPOUT_PARAM_ID, ''))
                    dropouts.insert(idx, v)
                    
        return dropouts

    def getOptimizerParams(self):
        return {k: v for k, v in self.bestParams.items() if k in OPTIMIZER_PARAMS} if self.wasTuned() else {}

    def getBatchSize(self):
        return DEFAULT_BATCH_SIZE if not self.wasTuned() else self.bestParams['batchSize']

    def getSaveFilePath(self, debug = DEBUG):
        return self.getWeightsFilePath() if not debug else DEBUG_FILE_PATH

    def wasTuned(self):
        return os.path.isfile(self.getParamFilePath())

    def update(self):
        if self.wasTuned():
            with open(self.getParamFilePath(), 'rb') as paramFile:
                trials = pickle.load(paramFile)
                best = trials.best_trial['misc']['vals']

                for k, v in best.items():
                    if type(v) is list:
                        best[k] = v[0]

                self.bestParams = getBestParams(self.getParamSpace(), best)

    def compile(self, params = {}, loss = LOSS, metrics = METRICS):
        if len(params) == 0 and self.bestParams: params = self.bestParams
        self.model.compile(loss = loss, optimizer = OPTIMIZER(**params), metrics = metrics)

    def tune(self, posDatasetFilePath, negDatasetFilePath, paths, labels, metric, verbose = True):
        paramSpace = self.getParamSpace()
        paramFilePath = self.getParamFilePath()

        best, trials = tune(paramSpace, self, posDatasetFilePath, negDatasetFilePath, paths, labels, metric, verbose = verbose)

        if verbose:
            print('Best model parameters found:', getBestParams(paramSpace, best))

        with open(paramFilePath, 'wb') as modelParamFile:
            pickle.dump(trials, modelParamFile)

    def _multiInputGenerator(self, generator):
        from data import SCALES

        while True:
                X, y = generator.next()
                X_extended = []

                for i in np.arange(0, self.stageIdx+1):
                    X_extended.append(np.vstack([cv2.resize(img, SCALES[i])[np.newaxis] for img in X]))

                yield X_extended, y

    def _getInputGenerator(self, generator):
        return generator if self.stageIdx == 0 else self._multiInputGenerator(generator)

    def fit(self, X_train, X_test, y_train, y_test, normalizer, numEpochs = DEFAULT_NUM_EPOCHS, saveFilePath = None, batchSize = None, 
            compileParams = None, dropouts = None, verbose = True, debug = DEBUG):
        
        params = [compileParams or self.getOptimizerParams(), dropouts or self.getDropouts()]
        saveFilePath = saveFilePath or self.getSaveFilePath()
        batchSize = batchSize or self.getBatchSize()

        callbacks = [ModelCheckpoint(saveFilePath, monitor = 'val_loss', save_best_only = True, verbose = int(verbose))]

        model = self(*params)
        y_train, y_test = (np_utils.to_categorical(vec, int(np.amax(vec) + 1)) for vec in (y_train, y_test))

        trainGenerator = normalizer.preprocess(X_train, labels = y_train, batchSize = batchSize, shuffle = True)
        validationGenerator = normalizer.preprocess(X_test, labels = y_test, batchSize = batchSize, shuffle = True, useDataAugmentation = False)

        if debug: print(params, saveFilePath, batchSize, X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        model.fit_generator(self._getInputGenerator(trainGenerator),
                            len(X_train)//batchSize,
                            epochs = numEpochs,
                            verbose = 2 if verbose else 0,
                            callbacks = callbacks,
                            validation_data = self._getInputGenerator(validationGenerator),
                            validation_steps = len(X_test)//batchSize,
                            max_q_size = DEFAULT_Q_SIZE)

        del self.trainedModel
        self.trainedModel = None

    def loadModel(self, weightsFilePath = None):
        return load_model(self.getWeightsFilePath() if weightsFilePath is None else weightsFilePath)

    def predict(self, X, normalizer, weightsFilePath = None):
        if self.trainedModel is None:
            self.trainedModel = self.loadModel(weightsFilePath)

        y = np.zeros((0, NUM_CALIBRATOR_CATEGORIES if isinstance(self, ObjectCalibrator) else NUM_CLASSIFIER_CATEGORIES))

        if type(X) is not tuple:
            X = (X,)
        
        for i in np.arange(0, len(X[0]), PREDICTION_BATCH_SIZE):
            batches = []

            for X_input in X:
                batches.insert(0, normalizer.preprocess(X_input[i:min(len(X_input), i + PREDICTION_BATCH_SIZE)]))

            predictions = self.trainedModel.predict(batches, batch_size = PREDICTION_BATCH_SIZE)
            y = np.vstack((y, predictions))

        return y

    def eval(self, X_test, y_test, normalizer, metric, weightsFilePath = None, **metric_kwargs):
        y_pred = self.predict(X_test, normalizer, weightsFilePath)
        return metric(y_test, np.argmax(y_pred, axis = 1), **metric_kwargs)

class ObjectCalibrator(ObjectClassifier, ABC):
    LOSS = 'categorical_crossentropy'

    def __init__(self, stageIdx):
        super().__init__(stageIdx)

    @abstractmethod
    def __call__(self):
        pass


class StageOneClassifier(ObjectClassifier):
    def __init__(self):
        self.stageIdx = 0
        super().__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*2, includeTop = True, compile = True):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(16, (3, 3), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size = (3,3),strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(16, activation = 'relu')(flattened)
        finalDropout = Dropout(dropouts[1])(fullyConnectedLayer)

        if includeTop: 
            outputLayer = Dense(2, activation = 'softmax')(finalDropout)
        
        self.model = Model(inputs = inputLayer, outputs = outputLayer if includeTop else fullyConnectedLayer)

        if compile: self.compile(compileParams)

        return self.model

class StageTwoClassifier(ObjectClassifier):
    def __init__(self):
        self.stageIdx = 1
        super().__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*2, includeTop = True, compile = True):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(64, (5, 5), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size=(3,3), strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(128, activation = 'relu')(flattened)

        stageOne = StageOneClassifier()
        assert os.path.isfile(stageOne.getWeightsFilePath()), STAGE_ONE_NOT_TRAINED_ERROR
        stageOneModel = stageOne(includeTop = False, compile = False)
        trainedStageOne = stageOne.loadModel()

        for i, layer in enumerate(stageOneModel.layers):
            layer.set_weights(trainedStageOne.layers[i].get_weights())
            layer.trainable = False

        mergedFullyConnectedLayer = concatenate([fullyConnectedLayer, stageOneModel.output])
        finalDropout = Dropout(dropouts[1])(mergedFullyConnectedLayer)

        if includeTop:
            outputLayer = Dense(2, activation = 'softmax')(finalDropout)

        self.model = Model(inputs = [stageOneModel.input, inputLayer], outputs = outputLayer if includeTop else mergedFullyConnectedLayer)

        if compile: self.compile(compileParams)

        return self.model

class StageOneCalibrator(ObjectCalibrator):
    def __init__(self):
        self.stageIdx = 0
        super().__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*2):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(16, (3, 3), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size = (3,3), strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(128, activation = 'relu')(flattened)
        finalDropout = Dropout(dropouts[1])(fullyConnectedLayer)
        outputLayer = Dense(45, activation = 'softmax')(finalDropout)

        self.model = Model(inputs = inputLayer, outputs = outputLayer)
        self.compile(compileParams)
        return self.model

class StageTwoCalibrator(ObjectCalibrator):
    def __init__(self):
        self.stageIdx = 1
        super().__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*2):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(32, (5, 5), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size = (3,3), strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(64, activation = 'relu')(flattened)
        finalDropout = Dropout(dropouts[1])(fullyConnectedLayer)
        outputLayer = Dense(45, activation = 'softmax')(finalDropout)

        self.model = Model(inputs = inputLayer, outputs = outputLayer)
        self.compile(compileParams)
        return self.model

MODELS = {False: [StageOneClassifier(), StageTwoClassifier()], True: [StageOneCalibrator(), StageTwoCalibrator()]}
