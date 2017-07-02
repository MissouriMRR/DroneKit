import h5py
import numpy as np
import atexit

from sklearn.model_selection import StratifiedKFold

from data import DATASET_LABEL, LABELS_LABEL, RANDOM_SEED
from util import TempH5pyFile

DEFAULT_NUM_FOLDS = 3
DEFAULT_PERCENT_TRAIN = .8

CONTEXT_MANAGER_NOT_USED_ERROR = 'You must instantiate the object with a context manager before calling this function'

class ClassifierDataset():
    def __init__(self, posDatasetFilePath, negDatasetFilePath = None, labels = None):
        self.posDatasetFilePath = posDatasetFilePath
        self.negDatasetFilePath = negDatasetFilePath
        self.labels = labels
        self.refCount = 0 

    def __enter__(self):
        self.refCount += 1

        datasetLen = 0
        useNegatives = self.negDatasetFilePath is not None

        self.positivesFile = h5py.File(self.posDatasetFilePath, 'r')
        self.positives = self.positivesFile[DATASET_LABEL]
        datasetLen += len(self.positives)

        if useNegatives:
            self.negativesFile = h5py.File(self.negDatasetFilePath, 'r')
            self.negatives = self.negativesFile[DATASET_LABEL]
            datasetLen += len(self.negatives)
        else:
            self.negatives = None

        self.trainingSet = TempH5pyFile('a').__enter__()
        self.trainingSet.create_dataset(DATASET_LABEL, (datasetLen, *self.positives.shape[1:]), dtype = self.positives.dtype)
        self.trainingSet[DATASET_LABEL][:len(self.positives)] = self.positives

        if useNegatives:
            self.trainingSet[DATASET_LABEL][len(self.positives):] = self.negatives

        atexit.register(self.__exit__)
        return self

    def __exit__(self, *args):
        if self.refCount > 0:
            for databaseFile in (self.positives, self.negatives, self.trainingSet):
                try:
                    databaseFile.close()
                except:
                    pass

        self.refCount -= 1

    def assertUsingContextManager(self):
        assert self.refCount == 1, CONTEXT_MANAGER_NOT_USED_ERROR

    def stratifiedSplitter(self, folds = DEFAULT_NUM_FOLDS, seed = RANDOM_SEED):
        self.assertUsingContextManager()

        skfolds = StratifiedKFold(n_splits = DEFAULT_NUM_FOLDS, random_state = seed)
        X = self.trainingSet[DATASET_LABEL]
        y = np.vstack((np.ones((len(self.positives), 1)), np.zeros((len(self.negatives), 1)))) if self.labels is None else self.labels

        for train_index, test_index in skfolds.split(np.zeros(len(y)), y.ravel()):
            X_train, X_test, y_train, y_test = (X[list(train_index)], X[list(test_index)], y[train_index], y[test_index])
            yield X_train, X_test, y_train, y_test

    def getStratifiedTrainingSet(self, percentTrain = DEFAULT_PERCENT_TRAIN, seed = RANDOM_SEED):
        numFolds = 1./(1-percentTrain)
        X_train, X_test, y_train, y_test = next(self.stratifiedSplitter(numFolds, seed))
        return X_train, X_test, y_train, y_test
