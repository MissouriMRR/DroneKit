from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.metrics import f1_score

from .dataset import ClassifierDataset
from .model import DEFAULT_NUM_EPOCHS
from .google_storage import upload

def train(model, datasetManager, numEpochs = DEFAULT_NUM_EPOCHS, tune = True):
    labels = datasetManager.getLabels()
    paths = datasetManager.getPaths()

    if not model.wasTuned() and tune:
        model.tune(datasetManager, labels, f1_score)

    with ClassifierDataset(paths[0], paths[1], labels) as dataset:
        X_train, X_test, y_train, y_test = dataset.getStratifiedTrainingSet()
        model.fit(X_train, X_test, y_train, y_test, datasetManager, numEpochs = numEpochs)
    
    upload(model.getWeightsFilePath())