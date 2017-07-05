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
DEFAULT_Q_SIZE = PREDICTION_BATCH_SIZE
DEBUG_FILE_PATH = 'debug.hdf'
PARAM_FILE_NAME_FORMAT_STR = '%sparams'

STAGE_ONE_NOT_TRAINED_ERROR = 'You must train the stage one models before moving onto stage two!'

NUM_CLASSIFIER_CATEGORIES = 2
NUM_CALIBRATOR_CATEGORIES = 45

class ObjectClassifier(ABC):
    DEFAULT_DROPOUT = .3
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

    def __init__(self, stageIdx):
        self.imgSize = SCALES[stageIdx]
        self.inputShape = (self.imgSize[1], self.imgSize[0], 3)
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
        from detect import NET_FILE_NAMES
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

    def _secondaryInputGenerator(self, generators):
        while True:
                X, y = generators[0].next()
                X_extended = [generator.next()[0] for generator in generators[1:]]
                X_extended.append(X)
                yield X_extended, y

    def _getInputGenerator(self, generators):
        return generators[0] if len(generators) == 1 else self._secondaryInputGenerator(generators)

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

        if type(X_train) is not tuple:
            X_train = (X_train,)
        if type(X_test) is not tuple:
            X_test = (X_test,)

        trainGenerators = tuple((normalizer.preprocess(X, labels = y_train, batchSize = batchSize, shuffle = True) for X in X_train))
        validationGenerators = tuple((normalizer.preprocess(X, labels = y_test, batchSize = batchSize, shuffle = True, useDataAugmentation = False) for X in X_test))

        model.fit_generator(self._getInputGenerator(trainGenerators),
                            len(X_train[0])//batchSize,
                            epochs = numEpochs,
                            verbose = 2 if verbose else 0,
                            callbacks = callbacks,
                            validation_data = self._getInputGenerator(validationGenerators),
                            validation_steps = len(X_test[0])//batchSize,
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
    PARAM_SPACE = StageOneClassifier.PARAM_SPACE

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

    def _passSecondaryInputs(self, X, callback, *args, **kwargs):
        resizeTo = SCALES[self.stageIdx - 1]

        with TempH5pyFile('a') as secondaryTrainInput, TempH5pyFile('a') as secondaryTestInput:
            for i, file in enumerate((secondaryTrainInput, secondaryTestInput)):
                if i < len(X) and type(X[i]) is not tuple:
                    file.create_dataset(DATASET_LABEL, (len(X[i]), *resizeTo, 3), dtype = X[i].dtype)
                    dataset = file[DATASET_LABEL]

                    for j in np.arange(len(X[i])):
                        dataset[j] = cv2.resize(X[i][j], resizeTo)

                    X[i] = (X[i], dataset)

            return callback(*X, *args, **kwargs)

    def fit(self, X_train, X_test, *args, **kwargs):
        self._passSecondaryInputs([X_train, X_test], super().fit, *args, **kwargs)

    def eval(self, X_test, *args, **kwargs):
        return self._passSecondaryInputs([X_test], super().eval, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self._passSecondaryInputs([X], super().predict, *args, **kwargs)

class StageOneCalibrator(ObjectCalibrator):
    PARAM_SPACE = StageOneClassifier.PARAM_SPACE

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
    PARAM_SPACE = StageOneClassifier.PARAM_SPACE

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
