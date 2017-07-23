from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pickle
import os
import cv2
import numpy as np
import atexit
import copy
import time
import six
import gc
import abc
from abc import abstractmethod
from collections import Iterable

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from keras.engine.topology import InputLayer
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from .hyperopt_keras import HyperoptWrapper, getBestParams, DEFAULT_NUM_EVALS, tune
from .google_storage import upload, downloadIfAvailable
from .task import DEBUG
from .preprocess import ImageNormalizer
from .data import SCALES, DATASET_LABEL, DatasetManager
from .util import TempH5pyFile
hp = HyperoptWrapper()

NET_FILE_NAMES = {False: {SCALES[0][0]: '12net.hdf', SCALES[1][0]: '24net.hdf', SCALES[2][0]: '48net.hdf'}, 
                  True: {SCALES[0][0]: '12calibnet.hdf', SCALES[1][0]: '24calibnet.hdf', SCALES[2][0]: '48calibnet.hdf'}}

DROPOUT_PARAM_ID = 'dropout'
OPTIMIZER_PARAMS = ['lr', 'momentum', 'decay', 'nesterov']
NORMALIZATION_PARAMS = ['norm', 'flip']
TRAIN_PARAMS = ['batchSize']

OPTIMIZER = SGD

DEFAULT_BATCH_SIZE = 128
PREDICTION_BATCH_SIZE = 256
DEFAULT_NUM_EPOCHS = 300
DEFAULT_Q_SIZE =  10
DEBUG_FILE_PATH = 'debug.hdf'
PARAM_FILE_NAME_FORMAT_STR = '%sparams'

STAGE_ONE_NOT_TRAINED_ERROR = 'You must train the stage one models before moving onto stage two!'
STAGE_TWO_NOT_TRAINED_ERROR = 'You must train the stage two models before moving onto stage three!'

def _convert(data):
    if isinstance(data, bytes):
        return data.decode('ascii')
    elif isinstance(data, (dict, tuple, list, set)):
        return type(data)(map(_convert, data.items() if hasattr(data, 'items') else data))

    return data

@six.add_metaclass(abc.ABCMeta)
class ObjectClassifier():
    DEFAULT_DROPOUT = .3
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

    def __init__(self, stageIdx):
        self.imgSize = SCALES[stageIdx]
        self.inputShape = (self.imgSize[1], self.imgSize[0], 3)
        self.additionalNormalizers = []
        self.trainedModel = None
        self.bestParams = None

        if stageIdx == 2:
            self.PARAM_SPACE = copy.deepcopy(self.PARAM_SPACE)
            self.PARAM_SPACE.update({'dropout2': hp.uniform(0, .75)})

        self.update()

    @abstractmethod
    def __call__(self):
        pass

    def getStageIdx(self):
        return self.stageIdx
    
    def getParamFilePath(self):
        paramFilePath = PARAM_FILE_NAME_FORMAT_STR % (os.path.splitext(self.getWeightsFilePath())[0],)
        if not os.path.isfile(paramFilePath): downloadIfAvailable(paramFilePath)
        return paramFilePath

    def getParamSpace(self):
        return self.PARAM_SPACE

    def getWeightsFilePath(self):
        weightsFilePath = NET_FILE_NAMES[isinstance(self, ObjectCalibrator)][SCALES[self.stageIdx][0]]
        if not os.path.isfile(weightsFilePath): downloadIfAvailable(weightsFilePath)
        return weightsFilePath

    def getNormalizationMethod(self):
        return self.bestParams['norm'] if self.wasTuned() else ImageNormalizer.STANDARD_NORMALIZATION

    def getNormalizationParams(self):
        params = {}

        if self.wasTuned():
            params = {k: v for k, v in self.bestParams.items() if k in NORMALIZATION_PARAMS}
            del params['norm']

        return params

    def getAdditionalNormalizers(self, datasetManagerParams = None):
        if not self.additionalNormalizers:
            for i in np.arange(0, self.stageIdx):
                self.additionalNormalizers.append(DatasetManager(MODELS[i][0], **datasetManagerParams).getNormalizer())

        return self.additionalNormalizers

    def getDropouts(self):
        dropouts = [self.DEFAULT_DROPOUT] * len([k for k in self.getParamSpace() if k.startswith(DROPOUT_PARAM_ID)])

        if self.wasTuned():
            dropouts = []
            for k, v in self.bestParams.items():
                if k.startswith(DROPOUT_PARAM_ID):
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
                loadArgs = {} if six.PY2 else {'encoding': 'bytes'}
                trials = pickle.load(paramFile, **loadArgs)
                trials.__dict__  = _convert(trials.__dict__)
                best = _convert(trials.best_trial['misc']['vals'])
                self.bestParams = getBestParams(self.getParamSpace(), best)

    def compile(self, params = {}, loss = LOSS, metrics = METRICS):
        if len(params) == 0 and self.bestParams: params = self.bestParams
        self.model.compile(loss = loss, optimizer = OPTIMIZER(**params), metrics = metrics)

    def tune(self, datasetManager, labels, metric, verbose = True, numEvals = DEFAULT_NUM_EVALS):
        paramSpace = self.getParamSpace()
        paramFilePath = self.getParamFilePath()

        best, trials = tune(paramSpace, self, datasetManager, labels, metric, verbose = verbose, numEvals = numEvals)

        if verbose:
            print('Best model parameters found:', getBestParams(paramSpace, best))

        with open(paramFilePath, 'wb') as modelParamFile:
            pickle.dump(trials, modelParamFile, protocol = 2)

        upload(paramFilePath)
        self.update()

    def getInputGenerator(self, X, y, normalizers, **normalizerParams):
        isCalibrationNet = isinstance(self, ObjectCalibrator)
        normalizerParams.update({'labels': y})
        generator = normalizers[self.stageIdx if not isCalibrationNet else 0].preprocess(X, **normalizerParams)
        normalizerParams.update({'batchSize': None, 'labels': None})

        while True:
                X, y = next(generator)
                X_extended = []
                
                if not isCalibrationNet:
                    for i in np.arange(0, self.stageIdx+1):
                        if i != self.stageIdx:
                            X_extended.append(normalizers[i].preprocess(np.vstack([cv2.resize(img, SCALES[i])[np.newaxis] for img in X]), **normalizerParams))

                X_extended.insert(self.stageIdx, X)

                yield X_extended, y

    def fit(self, X_train, X_test, y_train, y_test, datasetManager, normalizer = None, numEpochs = DEFAULT_NUM_EPOCHS, saveFilePath = None, batchSize = None, 
            compileParams = None, dropouts = None, verbose = True, debug = DEBUG):
        
        params = [compileParams or self.getOptimizerParams(), dropouts or self.getDropouts(), datasetManager.getParams()]
        saveFilePath = saveFilePath or self.getSaveFilePath(debug)
        batchSize = batchSize or self.getBatchSize()

        callbacks = [ModelCheckpoint(saveFilePath, monitor = 'val_loss', save_best_only = True, verbose = int(verbose))]

        model = self(*params)
        y_train, y_test = (np_utils.to_categorical(vec, int(np.amax(vec) + 1)) for vec in (y_train, y_test))

        normalizers = self.additionalNormalizers + [normalizer or datasetManager.getNormalizer()]

        if debug: print(params, saveFilePath, batchSize, X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        model.fit_generator(self.getInputGenerator(X_train, y_train, normalizers, shuffle = True, batchSize = batchSize),
                            len(X_train)//batchSize,
                            epochs = numEpochs,
                            verbose = 2 if verbose else 0,
                            callbacks = callbacks,
                            validation_data = self.getInputGenerator(X_test, y_test, normalizers, batchSize = batchSize, useDataAugmentation = False, shuffle = False),
                            validation_steps = len(X_test)//batchSize,
                            max_queue_size = DEFAULT_Q_SIZE)

        del self.trainedModel
        self.trainedModel = None
        K.clear_session()
        gc.collect()

    def loadModel(self, weightsFilePath = None):
        if self.trainedModel is None:
            self.trainedModel = load_model(self.getWeightsFilePath() if weightsFilePath is None else weightsFilePath)
            inputLayers = []

            for layer in self.trainedModel.layers:
                if type(layer) is InputLayer:
                    inputLayers.append(layer.input)

            inputLayers.append(K.learning_phase())
            self.predictFunc = K.function(inputLayers, [self.trainedModel.layers[-1].output])
        
        return self.trainedModel

    def predict(self, X, normalizer = None, weightsFilePath = None, datasetManager = None):
        self.loadModel(weightsFilePath)
        useFastPredict = datasetManager is None
        makePrediction = lambda X: self.predictFunc(X)[0]

        if not useFastPredict:
            batchSize = PREDICTION_BATCH_SIZE
            normalizers = self.getAdditionalNormalizers(datasetManager.getParams()) + [normalizer]
            inputGenerator = self.getInputGenerator(X, None, normalizers, batchSize = batchSize, shuffle = False, useDataAugmentation = False)
            
            for i in np.arange(0, len(X), PREDICTION_BATCH_SIZE):
                X_batch, y_batch = next(inputGenerator)
                batches = []

                for j, inputArray in enumerate(X_batch):
                    arraySlice = inputArray[:min(PREDICTION_BATCH_SIZE, len(X) - i)]

                    if j < 2:
                        batches.insert(0, arraySlice)
                    else:
                        batches.append(arraySlice)
                
                batches.append(0)
                predictions = makePrediction(batches)
                
                if i == 0: 
                    y = np.zeros((0, predictions.shape[1]))
                
                y = np.vstack((y, predictions))
        else:
            y = makePrediction(X + [0])
        
        return y

    def eval(self, X_test, y_test, normalizer, metric, weightsFilePath = None, datasetManager = None, **metric_kwargs):
        y_pred = self.predict(X_test, normalizer, weightsFilePath, datasetManager = datasetManager)
        return metric(y_test, np.argmax(y_pred, axis = 1), **metric_kwargs)

@six.add_metaclass(abc.ABCMeta)
class ObjectCalibrator(ObjectClassifier):
    LOSS = 'categorical_crossentropy'

    def __init__(self, stageIdx):
        super(ObjectCalibrator, self).__init__(stageIdx)

    @abstractmethod
    def __call__(self):
        pass


class StageOneClassifier(ObjectClassifier):
    HP = HyperoptWrapper()
    PARAM_SPACE = {
        'dropout0': HP.uniform(0, .75),
        'dropout1': HP.uniform(0, .75),
        'lr': HP.loguniform(1e-4, 1),
        'batchSize': HP.choice(128),
        'norm':  HP.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': HP.choice(ImageNormalizer.FLIP_HORIZONTAL),
        'momentum': HP.choice(.9),
        'decay': HP.choice(1e-4),
        'nesterov': HP.choice(True)
    }

    def __init__(self):
        self.stageIdx = 0
        super(StageOneClassifier, self).__init__(self.stageIdx)
    
    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*2, datasetManagerParams = {}, includeTop = True, compile = True):
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
    HP = HyperoptWrapper()
    PARAM_SPACE = {
        'dropout0': HP.uniform(0, .75),
        'dropout1': HP.uniform(0, .75),
        'lr': HP.loguniform(1e-4, 1),
        'batchSize': HP.choice(512),
        'norm':  HP.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': HP.choice(ImageNormalizer.FLIP_HORIZONTAL),
        'momentum': HP.choice(.9),
        'decay': HP.choice(1e-4),
        'nesterov': HP.choice(True)
    }

    def __init__(self):
        self.stageIdx = 1
        super(StageTwoClassifier, self).__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*2, datasetManagerParams = {}, includeTop = True, compile = True):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(64, (5, 5), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size=(3,3), strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(128, activation = 'relu')(flattened)

        stageOne = StageOneClassifier()
        assert os.path.isfile(stageOne.getWeightsFilePath()), STAGE_ONE_NOT_TRAINED_ERROR
        stageOneModel = stageOne(datasetManagerParams = datasetManagerParams, includeTop = False, compile = False)
        trainedStageOne = stageOne.loadModel()

        for i, layer in enumerate(stageOneModel.layers):
            layer.set_weights(trainedStageOne.layers[i].get_weights())
            layer.trainable = False
        
        if not self.additionalNormalizers:
            self.additionalNormalizers.append(DatasetManager(stageOne, **datasetManagerParams).getNormalizer())
        
        mergedFullyConnectedLayer = concatenate([fullyConnectedLayer, stageOneModel.output])
        finalDropout = Dropout(dropouts[1])(mergedFullyConnectedLayer)

        if includeTop:
            outputLayer = Dense(2, activation = 'softmax')(finalDropout)

        self.model = Model(inputs = [stageOneModel.input, inputLayer], outputs = outputLayer if includeTop else mergedFullyConnectedLayer)

        if compile: self.compile(compileParams)

        return self.model

class StageThreeClassifier(ObjectClassifier):
    # temporary patch for HyperoptWrapper bug. need to change HyperoptWrapper class and retune everything to fix
    HP = HyperoptWrapper()
    PARAM_SPACE = {
        'dropout0': HP.uniform(0, .75),
        'dropout1': HP.uniform(0, .75),
        'lr': HP.loguniform(1e-4, 1),
        'batchSize': HP.choice(512),
        'norm':  HP.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': HP.choice(ImageNormalizer.FLIP_HORIZONTAL),
        'momentum': HP.choice(.9),
        'decay': HP.choice(1e-4),
        'nesterov': HP.choice(True)
    }

    def __init__(self):
        self.stageIdx = 2
        super(StageThreeClassifier, self).__init__(self.stageIdx)
    
    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*3, datasetManagerParams = {}):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(64, (5, 5), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size=(3,3), strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)
        firstBatchNorm = BatchNormalization()(firstDropout)
        secondaryConv2D = Conv2D(64, (5, 5), activation = 'relu')(firstBatchNorm)
        secondaryBatchNorm = BatchNormalization()(secondaryConv2D)
        secondaryMaxPool2D = MaxPooling2D(pool_size=(3,3), strides = 2)(secondaryBatchNorm)
        secondDropout = Dropout(dropouts[1])(secondaryMaxPool2D)

        flattened = Flatten()(firstDropout)
        fullyConnectedLayer = Dense(256, activation = 'relu')(flattened)

        stageTwo = StageTwoClassifier()
        assert os.path.isfile(stageTwo.getWeightsFilePath()), STAGE_TWO_NOT_TRAINED_ERROR
        trainedStageTwo = stageTwo.loadModel()
        stageTwoModel = stageTwo(datasetManagerParams = datasetManagerParams, includeTop = False, compile = False)
        inputLayers = [inputLayer]

        for i, layer in enumerate(stageTwoModel.layers):
            layer.set_weights(trainedStageTwo.layers[i].get_weights())
            layer.trainable = False

            if type(layer) is InputLayer:
                inputLayers.insert(0, layer.input)
        
        if not self.additionalNormalizers:
            self.additionalNormalizers.extend(stageTwo.getAdditionalNormalizers())
            self.additionalNormalizers.append(DatasetManager(stageTwo, **datasetManagerParams).getNormalizer())

        mergedFullyConnectedLayer = concatenate([fullyConnectedLayer, stageTwoModel.output])
        thirdDropout = Dropout(dropouts[2])(mergedFullyConnectedLayer)
        outputLayer = Dense(2, activation = 'softmax')(thirdDropout)

        self.model = Model(inputs = inputLayers, outputs = outputLayer)
        self.compile(compileParams)
        return self.model

class StageOneCalibrator(ObjectCalibrator):
    HP = HyperoptWrapper()
    PARAM_SPACE = {
        'dropout0': HP.uniform(0, .75),
        'dropout1': HP.uniform(0, .75),
        'lr': HP.loguniform(1e-4, 1),
        'batchSize': HP.choice(512),
        'norm':  HP.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': HP.choice(None),
        'momentum': HP.choice(.9),
        'decay': HP.choice(1e-4),
        'nesterov': HP.choice(True)
    }

    def __init__(self):
        self.stageIdx = 0
        super(StageOneCalibrator, self).__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*2, datasetManagerParams = {}):
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
    HP = HyperoptWrapper()
    PARAM_SPACE = {
        'dropout0': HP.uniform(0, .75),
        'dropout1': HP.uniform(0, .75),
        'lr': HP.loguniform(1e-4, 1),
        'batchSize': HP.choice(512),
        'norm':  HP.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': HP.choice(None),
        'momentum': HP.choice(.9),
        'decay': HP.choice(1e-4),
        'nesterov': HP.choice(True)
    }

    def __init__(self):
        self.stageIdx = 1
        super(StageTwoCalibrator, self).__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*2, datasetManagerParams = {}):
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

class StageThreeCalibrator(ObjectCalibrator):
    #TODO: Fix HyperoptWrapper class
    HP = HyperoptWrapper()
    PARAM_SPACE = {
        'dropout0': HP.uniform(0, .75),
        'dropout1': HP.uniform(0, .75),
        'lr': HP.loguniform(1e-9, 1),
        'batchSize': HP.choice(512),
        'norm':  HP.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': HP.choice(None),
        'momentum': HP.choice(.9),
        'decay': HP.choice(1e-4),
        'nesterov': HP.choice(True)
    }

    def __init__(self):
        self.stageIdx = 2
        super(StageThreeCalibrator, self).__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*3, datasetManagerParams = {}):
        inputLayer = Input(shape = self.inputShape)
        conv2D = Conv2D(64, (5, 5), activation = 'relu')(inputLayer)
        maxPool2D = MaxPooling2D(pool_size = (3,3), strides = 2)(conv2D)
        firstDropout = Dropout(dropouts[0])(maxPool2D)
        firstBatchNorm = BatchNormalization()(firstDropout)
        secondaryConv2D = Conv2D(64, (5, 5), activation = 'relu')(firstBatchNorm)
        secondaryMaxPool2D = MaxPooling2D(pool_size = (3,3), strides = 2)(secondaryConv2D)
        secondBatchNorm = BatchNormalization()(secondaryMaxPool2D)
        secondDropout = Dropout(dropouts[1])(secondBatchNorm)

        flattened = Flatten()(secondDropout)
        fullyConnectedLayer = Dense(256, activation = 'relu')(flattened)
        thirdDropout = Dropout(dropouts[2])(fullyConnectedLayer)
        outputLayer = Dense(45, activation = 'softmax')(thirdDropout)

        self.model = Model(inputs = inputLayer, outputs = outputLayer)
        self.compile(compileParams)
        return self.model

MODELS = {False: [StageOneClassifier(), StageTwoClassifier(), StageThreeClassifier()], True: [StageOneCalibrator(), StageTwoCalibrator(), StageThreeCalibrator()]}