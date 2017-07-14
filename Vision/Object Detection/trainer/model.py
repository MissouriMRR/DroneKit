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
import abc
from abc import abstractmethod

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from keras.engine.topology import InputLayer
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from .hyperopt_keras import HyperoptWrapper, getBestParams, DEFAULT_NUM_EVALS, tune
from .google_storage import upload
from .task import DEBUG
from .preprocess import ImageNormalizer
from .data import SCALES, DATASET_LABEL, FACE_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS
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
PREDICTION_BATCH_SIZE = 1024
DEFAULT_NUM_EPOCHS = 300
DEFAULT_Q_SIZE = 5000
DEBUG_FILE_PATH = 'debug.hdf'
PARAM_FILE_NAME_FORMAT_STR = '%sparams'

STAGE_ONE_NOT_TRAINED_ERROR = 'You must train the stage one models before moving onto stage two!'
STAGE_TWO_NOT_TRAINED_ERROR = 'You must train the stage two models before moving onto stage three!'

# sauce: https://stackoverflow.com/questions/33137741/fastest-way-to-convert-a-dicts-keys-values-from-bytes-to-str-in-python3
def _convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(_convert, data.items()))
    if isinstance(data, tuple):  return map(_convert, data)
    if isinstance(data, tuple):  return tuple(map(_convert, data))
    if isinstance(data, list):   return list(map(_convert, data))
    if isinstance(data, set):    return set(map(_convert, data))
    return data

@six.add_metaclass(abc.ABCMeta)
class ObjectClassifier():
    PARAM_SPACE = {
        'dropout0': hp.uniform(0, .75),
        'dropout1': hp.uniform(0, .75),
        'lr': hp.loguniform(1e-4, .3),
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
    
    def getParamFilePath(self):
        return PARAM_FILE_NAME_FORMAT_STR % (os.path.splitext(self.getWeightsFilePath())[0],)

    def getParamSpace(self):
        return self.PARAM_SPACE

    def getWeightsFilePath(self):
        return NET_FILE_NAMES[isinstance(self, ObjectCalibrator)][SCALES[self.stageIdx][0]]

    def getNormalizationMethod(self):
        return self.bestParams['norm'] if self.wasTuned() else ImageNormalizer.STANDARD_NORMALIZATION

    def getNormalizationParams(self):
        params = {}

        if self.wasTuned():
            params = {k: v for k, v in self.bestParams.items() if k in NORMALIZATION_PARAMS}
            del params['norm']

        return params

    def getAdditionalNormalizers(self):
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
                trials = pickle.load(paramFile, encoding='bytes')
                trials.__dict__  = _convert(trials.__dict__)
                best = _convert(trials.best_trial['misc']['vals'])
                self.bestParams = getBestParams(self.getParamSpace(), best)

    def compile(self, params = {}, loss = LOSS, metrics = METRICS):
        if len(params) == 0 and self.bestParams: params = self.bestParams
        self.model.compile(loss = loss, optimizer = OPTIMIZER(**params), metrics = metrics)

    def tune(self, posDatasetFilePath, negDatasetFilePath, paths, labels, metric, verbose = True, numEvals = DEFAULT_NUM_EVALS):
        paramSpace = self.getParamSpace()
        paramFilePath = self.getParamFilePath()

        best, trials = tune(paramSpace, self, posDatasetFilePath, negDatasetFilePath, paths, labels, metric, verbose = verbose, numEvals = numEvals)

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

    def fit(self, X_train, X_test, y_train, y_test, normalizer, numEpochs = DEFAULT_NUM_EPOCHS, saveFilePath = None, batchSize = None, 
            compileParams = None, dropouts = None, verbose = True, debug = DEBUG):
        
        params = [compileParams or self.getOptimizerParams(), dropouts or self.getDropouts()]
        saveFilePath = saveFilePath or self.getSaveFilePath(debug)
        batchSize = batchSize or self.getBatchSize()

        callbacks = [ModelCheckpoint(saveFilePath, monitor = 'val_loss', save_best_only = True, verbose = int(verbose))]

        model = self(*params)
        y_train, y_test = (np_utils.to_categorical(vec, int(np.amax(vec) + 1)) for vec in (y_train, y_test))

        normalizers = self.additionalNormalizers + [normalizer]

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

    def predict(self, X, normalizer = None, weightsFilePath = None):
        self.loadModel(weightsFilePath)
        useFastPredict = normalizer is None
        makePrediction = lambda X: self.predictFunc(X)[0]

        if not useFastPredict:
            batchSize = PREDICTION_BATCH_SIZE
            normalizers = self.additionalNormalizers + [normalizer]
            inputGenerator = self.getInputGenerator(X, None, normalizers, batchSize = batchSize, shuffle = False, useDataAugmentation = False)
            
            for i in np.arange(0, len(X), PREDICTION_BATCH_SIZE):
                X_batch, y_batch = next(inputGenerator)
                batches = []

                for inputArray in X_batch:
                    batches.insert(0, inputArray[:min(PREDICTION_BATCH_SIZE, len(X) - i)])
                
                batches.append(0)
                predictions = makePrediction(batches)
                
                if i == 0: 
                    y = np.zeros((0, predictions.shape[1]))
                
                y = np.vstack((y, predictions))
        else:
            X.append(0)
            y = makePrediction(X)
        
        return y

    def eval(self, X_test, y_test, normalizer, metric, weightsFilePath = None, **metric_kwargs):
        y_pred = self.predict(X_test, normalizer, weightsFilePath)
        return metric(y_test, np.argmax(y_pred, axis = 1), **metric_kwargs)

@six.add_metaclass(abc.ABCMeta)
class ObjectCalibrator(ObjectClassifier):
    PARAM_SPACE = {
        'dropout0': hp.uniform(0, .75),
        'dropout1': hp.uniform(0, .75),
        'lr': hp.loguniform(1e-4, .3),
        'batchSize': hp.choice(32, 64, 128, 256),
        'norm':  hp.choice(ImageNormalizer.STANDARD_NORMALIZATION),
        'flip': hp.choice(None)
    }

    LOSS = 'categorical_crossentropy'

    def __init__(self, stageIdx):
        super(ObjectCalibrator, self).__init__(stageIdx)

    @abstractmethod
    def __call__(self):
        pass


class StageOneClassifier(ObjectClassifier):
    def __init__(self):
        self.stageIdx = 0
        super(StageOneClassifier, self).__init__(self.stageIdx)
    
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
        super(StageTwoClassifier, self).__init__(self.stageIdx)

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

        self.additionalNormalizers.append(ImageNormalizer(FACE_DATABASE_PATHS[self.stageIdx - 1], NEGATIVE_DATABASE_PATHS[self.stageIdx - 1], 
                                                          stageOne.getNormalizationMethod()))
        
        mergedFullyConnectedLayer = concatenate([fullyConnectedLayer, stageOneModel.output])
        finalDropout = Dropout(dropouts[1])(mergedFullyConnectedLayer)

        if includeTop:
            outputLayer = Dense(2, activation = 'softmax')(finalDropout)

        self.model = Model(inputs = [stageOneModel.input, inputLayer], outputs = outputLayer if includeTop else mergedFullyConnectedLayer)

        if compile: self.compile(compileParams)

        return self.model

class StageThreeClassifier(ObjectClassifier):
    def __init__(self):
        self.stageIdx = 2
        super(StageThreeClassifier, self).__init__(self.stageIdx)
    
    def __call__(self, compileParams = {}, dropouts = [ObjectClassifier.DEFAULT_DROPOUT]*3):
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
        stageTwoModel = stageTwo(includeTop = False, compile = False)
        inputLayers = [inputLayer]

        for i, layer in enumerate(stageTwoModel.layers):
            layer.set_weights(trainedStageTwo.layers[i].get_weights())
            layer.trainable = False

            if type(layer) is InputLayer:
                inputLayers.insert(0, layer.input)
        
        self.additionalNormalizers.extend(stageTwo.getAdditionalNormalizers())
        self.additionalNormalizers.append(ImageNormalizer(FACE_DATABASE_PATHS[self.stageIdx - 1], NEGATIVE_DATABASE_PATHS[self.stageIdx - 1],
                                                          stageTwo.getNormalizationMethod()))

        mergedFullyConnectedLayer = concatenate([fullyConnectedLayer, stageTwoModel.output])
        thirdDropout = Dropout(dropouts[2])(mergedFullyConnectedLayer)
        outputLayer = Dense(2, activation = 'softmax')(thirdDropout)

        self.model = Model(inputs = inputLayers, outputs = outputLayer)
        self.compile(compileParams)
        return self.model

class StageOneCalibrator(ObjectCalibrator):
    def __init__(self):
        self.stageIdx = 0
        super(StageOneCalibrator, self).__init__(self.stageIdx)

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
        super(StageTwoCalibrator, self).__init__(self.stageIdx)

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

class StageThreeCalibrator(ObjectCalibrator):
    def __init__(self):
        self.stageIdx = 2
        super(StageThreeCalibrator, self).__init__(self.stageIdx)

    def __call__(self, compileParams = {}, dropouts = [ObjectCalibrator.DEFAULT_DROPOUT]*3):
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

if __name__ == '__main__':
    from .dataset import ClassifierDataset
    model = StageOneClassifier()
    X_test_gen, y_test_gen = (np.zeros((0,) + (model.inputShape)), np.zeros((0, 1)))
    posDatasetFilePath, negDatasetFilePath = (FACE_DATABASE_PATHS[model.stageIdx], NEGATIVE_DATABASE_PATHS[model.stageIdx])
    
    yesNo = {True: 'yes', False: 'no'}

    with ClassifierDataset(posDatasetFilePath, negDatasetFilePath, None) as dataset:
        X_train, X_test, y_train, y_test = dataset.getStratifiedTrainingSet()
        numPositiveTrainSamples, numPositiveTestSamples = (len(np.where(vec == 1)[0]) for vec in (y_train, y_test))
        numNegativeTrainSamples, numNegativeTestSamples = (len(np.where(vec == 0)[0]) for vec in (y_train, y_test))
        print('Number of postive samples in train set:', numPositiveTrainSamples)
        print('Number of negative samples in train set:', numNegativeTrainSamples)
        print('Number of positive samples in test set:', numPositiveTestSamples)
        print('Number of negatives samples in test set:', numNegativeTestSamples)
        print('Percent positive in train set:', numPositiveTrainSamples/(numNegativeTrainSamples + numPositiveTrainSamples))
        print('Percent positive in test set:', numPositiveTestSamples/(numPositiveTestSamples + numNegativeTestSamples))
        print('Percent train:', len(y_train)/(len(y_test)+len(y_train)))
        print('Are the number of positive and negative samples consistent with the image array lengths? %s' % 
              yesNo.get((numPositiveTrainSamples + numNegativeTrainSamples) == len(X_train) and (numPositiveTestSamples + numNegativeTestSamples) == len(X_test)))
        print('Are label and image array lengths consistent? %s' % yesNo.get((len(X_test) == len(y_test) and len(X_train) == len(y_train))))
        
        normalizer = ImageNormalizer(posDatasetFilePath, negDatasetFilePath, model.getNormalizationMethod())
        normalizer.addDataAugmentationParams(model.getNormalizationParams())
        X_test_preprocessed = normalizer.preprocess(X_test, useDataAugmentation = False, shuffle = False)
        inputGenerator = model.getInputGenerator(X_test, y_test, (normalizer,), batchSize = PREDICTION_BATCH_SIZE, shuffle = False, useDataAugmentation = False)

        for i in np.arange(0, len(X_test), PREDICTION_BATCH_SIZE):
            X, y = next(inputGenerator)
            endIdx = min(PREDICTION_BATCH_SIZE, len(X_test) - i)
            X_test_gen = np.vstack((X_test_gen, X[0][:endIdx]))
            y_test_gen = np.vstack((y_test_gen, y[:endIdx]))

        print('Are the shapes of the arrays outputted by the input generator for the test set correct? %s' % 
              yesNo.get(X_test_gen.shape == X_test.shape and y_test_gen.shape == y_test.shape))
        print('Is input generator output for test set labels correct? %s' % yesNo.get(np.allclose(y_test_gen, y_test)))
        print('Is input generator output for test set images correct? %s' % yesNo.get(np.allclose(X_test_gen, X_test_preprocessed)))
        print('Training', type(model), ' with normalization method', model.getNormalizationMethod(), 'and augmentation params', model.getNormalizationParams())
        model.fit(X_train, X_test, y_train, y_test, normalizer, 5, debug = True)