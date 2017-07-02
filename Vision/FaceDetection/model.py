import pickle
import os
import multiprocessing
import numpy as np
import atexit
import copy
from abc import ABC, abstractmethod

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
atexit.register(K.clear_session)

from hyperopt_keras import HyperoptWrapper, getBestParams, parseParams, tune
from FaceDetection import DEBUG
from preprocess import ImageNormalizer
from detect import NET_FILE_NAMES
from data import SCALES
hp = HyperoptWrapper()

DROPOUT_PARAM_ID = 'dropout'
OPTIMIZER_PARAMS = ['lr', 'momentum', 'decay', 'nesterov']
NORMALIZATION_PARAMS = ['norm', 'flip']
TRAIN_PARAMS = ['batchSize']

OPTIMIZER = SGD

DEFAULT_BATCH_SIZE = 128
PREDICTION_BATCH_SIZE = 4096
DEFAULT_NUM_EPOCHS = 300
DEFAULT_Q_SIZE = 512
DEBUG_FILE_PATH = 'debug.hdf'
PARAM_FILE_NAME_FORMAT_STR = '%sparams'

NUM_CLASSIFIER_CATEGORIES = 2
NUM_CALIBRATOR_CATEGORIES = 45

class ObjectClassifier(ABC):
    DEFAULT_DROPOUT = .3
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

    def __init__(self, stageIdx):
        self.img_size = SCALES[stageIdx]
        self.input_shape = (self.img_size[1], self.img_size[0], 3)
        self.defaultParams = {}
        self.normalizationParams = None
        self.compileParams = None
        self.trainParams = None
        self.dropouts = None
        self.trainedModel = None

    @abstractmethod
    def __call__(self):
        pass

    def update(self):
        if len(self.defaultParams) == 0:
            if self.wasTuned():
                with open(self.getParamFilePath(), 'rb') as paramFile:
                    trials = pickle.load(paramFile)
                    best = trials.best_trial['misc']['vals']

                    for k, v in best.items():
                        if type(v) is list:
                            best[k] = v[0]

                    self.defaultParams.update(getBestParams(self.getParamSpace(), best))
                    self.normalizationParams, self.compileParams, self.trainParams, self.dropouts = parseParams(self.defaultParams)

    def compile(self, params = {}, loss = LOSS, metrics = METRICS):
        if len(params) == 0: params = self.defaultParams
        self.model.compile(loss = loss, optimizer = OPTIMIZER(**params), metrics = metrics)

    def getWeightsFilePath(self):
        return NET_FILE_NAMES[isinstance(self, ObjectCalibrator)][SCALES[self.stageIdx][0]]

    def getParamFilePath(self):
        return PARAM_FILE_NAME_FORMAT_STR % (os.path.splitext(self.getWeightsFilePath())[0],)

    def getParamSpace(self):
        return self.PARAM_SPACE

    def getNormalizationMethod(self):
        self.update()
        return self.normalizationParams['norm'] if self.normalizationParams.get('norm') is not None else ImageNormalizer.STANDARD_NORMALIZATION

    def getNormalizationParams(self):
        self.update()
        params = copy.deepcopy(self.normalizationParams)
        del params['norm']
        return params

    def wasTuned(self):
        return os.path.isfile(self.getParamFilePath())

    def tune(self, posDatasetFilePath, negDatasetFilePath, paths, labels, metric, verbose = True):
        paramSpace = self.getParamSpace()
        paramFilePath = self.getParamFilePath()

        best, trials = tune(paramSpace, self, posDatasetFilePath, negDatasetFilePath, paths, labels, metric, verbose = verbose)

        if verbose:
            print('Best model parameters found:', getBestParams(paramSpace, best))

        with open(paramFilePath, 'wb') as modelParamFile:
            pickle.dump(trials, modelParamFile)

    def fit(self, X_train, X_test, y_train, y_test, normalizer, numEpochs = DEFAULT_NUM_EPOCHS, saveFilePath = None, batchSize = None, 
            compileParams = None, dropouts = None, verbose = True, debug = DEBUG):
        self.update()
        params = {'compileParams': compileParams if compileParams is not None else self.compileParams, 
                  'dropouts': dropouts if dropouts is not None else self.dropouts}
        
        for k, v in list(params.items()):
            if v is None:
                del params[k]

        if saveFilePath is None:
            saveFilePath = self.getWeightsFilePath() if not debug else DEBUG_FILE_PATH
        if batchSize is None:
            batchSize = self.defaultParams['batchSize'] if self.defaultParams.get('batchSize') is not None else DEFAULT_BATCH_SIZE

        callbacks = [ModelCheckpoint(saveFilePath, monitor = 'val_loss', save_best_only = True, verbose = int(verbose))]

        model = self(**params)
        y_train, y_test = (np_utils.to_categorical(vec, int(np.amax(vec) + 1)) for vec in (y_train, y_test))
        trainGenerator = normalizer.preprocess(X_train, labels = y_train, batchSize = batchSize, shuffle = True)
        validationGenerator = normalizer.preprocess(X_test, labels = y_test, batchSize = batchSize, shuffle = True, useDataAugmentation = False)

        if debug:
            print(params, batchSize, saveFilePath, trainGenerator, validationGenerator)

        model.fit_generator(trainGenerator,
                            len(X_train)//batchSize,
                            numEpochs,
                            verbose = 2 if verbose else 0,
                            callbacks = callbacks,
                            validation_data = validationGenerator,
                            validation_steps = len(X_test)//batchSize,
                            max_q_size = DEFAULT_Q_SIZE,
                            workers = multiprocessing.cpu_count(),
                            pickle_safe = True)

        del self.trainedModel
        self.trainedModel = None

    def loadModel(self, weightsFilePath = None):
        return load_model(self.getWeightsFilePath() if weightsFilePath is None else weightsFilePath)

    def predict(self, X, normalizer):
        y = np.zeros((0, NUM_CALIBRATOR_CATEGORIES if isinstance(self, ObjectCalibrator) else NUM_CLASSIFIER_CATEGORIES))
        
        for i in np.arange(0, len(X), PREDICTION_BATCH_SIZE):
            batch = normalizer.preprocess(X[i:min(len(X), i + PREDICTION_BATCH_SIZE)])
            predictions = self.trainedModel.predict(batch, batch_size = PREDICTION_BATCH_SIZE)
            y = np.vstack((y, predictions))

        return y

    def eval(self, X_test, y_test, normalizer, metric, weightsFilePath = None, **metric_kwargs):
        if self.trainedModel is None:
            self.trainedModel = self.loadModel(weightsFilePath)

        y_pred = self.predict(X_test, normalizer)
        return metric(y_test, np.argmax(y_pred, axis = 1), **metric_kwargs)

class ObjectCalibrator(ObjectClassifier, ABC):
    LOSS = 'categorical_crossentropy'

    def __init__(self, stageIdx):
        super().__init__(stageIdx)

    @abstractmethod
    def __call__(self):
        pass


class StageOneClassifier(ObjectClassifier):
    PARAM_SPACE = {
        'dropout0': hp.uniform(0, .5),
        'dropout1': hp.uniform(0, .5),
        'lr': hp.loguniform(1e-4, .1),
        'batchSize': hp.choice(32, 64, 128, 256),
        'norm':  hp.choice(ImageNormalizer.STANDARD_NORMALIZATION, ImageNormalizer.MIN_MAX_SCALING),
        'flip': hp.choice(None, ImageNormalizer.FLIP_HORIZONTAL)
    }

    def __init__(self):
        self.stageIdx = 0
        super().__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*2, includeTop = True, compile = True):
        inputLayer = Input(shape = self.input_shape)
        conv2D = Conv2D(16, (3, 3), activation='relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size=(3,3),strides=2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(16, activation='relu')(flattened)
        finalDropout = Dropout(dropouts[1])(fullyConnectedLayer)

        if includeTop: 
            outputLayer = Dense(2, activation='softmax')(finalDropout)
        
        self.model = Model(inputs = inputLayer, outputs = outputLayer if includeTop else fullyConnectedLayer)

        if compile: self.compile(compileParams)

        return self.model

class StageOneCalibrator(ObjectCalibrator):
    PARAM_SPACE = StageOneClassifier.PARAM_SPACE

    def __init__(self):
        self.stageIdx = 0
        super().__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*2):
        inputLayer = Input(shape = self.input_shape)
        conv2D = Conv2D(16, (3, 3), activation='relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size=(3,3),strides=2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(128, activation='relu')(flattened)
        finalDropout = Dropout(dropouts[1])(fullyConnectedLayer)
        outputLayer = Dense(45, activation='softmax')(finalDropout)

        self.model = Model(inputs = inputLayer, outputs = outputLayer)
        self.compile(compileParams)
        return self.model


MODELS = {False: [StageOneClassifier()], True: [StageOneCalibrator()]}
