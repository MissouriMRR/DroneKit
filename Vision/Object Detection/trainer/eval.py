from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score

from .dataset import ClassifierDataset
from .model import ObjectCalibrator
from .detect import THRESHOLDS

class ModelEvaluator():
    def __init__(self, model, datasetManager):
        self.model = model
        self.datasetManager = datasetManager
        self.dataset = ClassifierDataset(*datasetManager.getPaths(), datasetManager.getLabels())
        self.isCalib = isinstance(model, ObjectCalibrator)
        self.average = 'macro' if self.isCalib else 'binary'
        self.X_test = None
        self.y_test = None

    def __enter__(self):
        try:
            self.dataset.__enter__()
        except:
            self.dataset.__exit__()
            raise
        
        X_train, X_test, y_train, y_test = self.dataset.getStratifiedTrainingSet()
        self.X_test, self.y_test = (X_test, y_test)
        self.predictions = self._predict()
        self.y_pred = np.argmax(self.predictions, axis = 1) if self.isCalib else (self.predictions[:, 1] >= THRESHOLDS[self.model.getStageIdx()])
        self.y_pred = self.y_pred.reshape(len(y_test), 1)
        return self

    def __exit__(self, *args):
        self.dataset.__exit__() 

    def _predict(self):
        return self.model.predict(self.X_test, self.datasetManager.getNormalizer(), datasetManager = self.datasetManager)

    def confusion_matrix(self, visualize = False):
        matrix = confusion_matrix(self.y_test, self.y_pred)
        
        if visualize:
            plt.matshow(matrix, cmap = plt.cm.gray)
            plt.show()

        return matrix

    def f1_score(self):
        return f1_score(self.y_test, self.y_pred, average = self.average)

    def accuracy(self):
        return np.sum(np.equal(self.y_test, self.y_pred), dtype=np.float64)/len(self.y_test)

    def precision(self):
        return precision_score(self.y_test, self.y_pred, average = self.average)

    def recall(self):
        return recall_score(self.y_test, self.y_pred, average = self.average)

    def roc_auc_score(self):
        return roc_auc_score(self.y_test, self.y_pred, average = None if self.average == 'binary' else self.average)
    
    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.predictions[:, 1])
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def plot_precision_recall_curve(self):
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, self.predictions[:, 1])
        plt.plot(recalls, precisions, 'b--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    def plot_precision_recall_vs_threshold(self):
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, self.predictions[:, 1])
        plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    def summary(self):
        print('\nPrecision:\t\t%.3f%%' % (self.precision()*100,))
        print('Recall:\t\t\t%.3f%%' % (self.recall()*100,))
        print('Accuracy:\t\t%.3f%%' % (self.accuracy()*100,))
        print('f1-score:\t\t%.3f' % self.f1_score())
        if not self.isCalib: print('Area under ROC curve:\t%.3f' % self.roc_auc_score())
        print('\nConfusion Matrix:\n', self.confusion_matrix())
        print()

        if self.isCalib:
            print('Plotting confusion matrix...')
            self.confusion_matrix(visualize = True)
        else:
            print('Plotting ROC curve...')
            self.plot_roc_curve()
            print('Plotting precision-recall curve...')
            self.plot_precision_recall_curve()
            print('Plotting precision and recall vs. threshold curve...')
            self.plot_precision_recall_vs_threshold()
