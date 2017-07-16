from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from .dataset import ClassifierDataset

def plot_precision_recall_vs_threshold(model, datasetManager):
    with ClassifierDataset(*datasetManager.getPaths()) as dataset:
        X_train, X_test, y_train, y_test = dataset.getStratifiedTrainingSet()
        precisions, recalls, thresholds = precision_recall_curve(y_test, model.predict(X_test, datasetManager.getNormalizer(), datasetManager = datasetManager)[:, 1])
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()