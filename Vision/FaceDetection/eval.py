import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from model import MODELS
from dataset import ClassifierDataset
from data import FACE_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS
from preprocess import ImageNormalizer

def plot_precision_recall_vs_threshold(stageIdx, isCalib):
    model = MODELS[isCalib][stageIdx]
    posDatasetFilePath = FACE_DATABASE_PATHS[stageIdx]
    negDatasetFilePath = NEGATIVE_DATABASE_PATHS[stageIdx]
    paths = [posDatasetFilePath, negDatasetFilePath]

    with ClassifierDataset(*paths) as dataset:
        model = MODELS[isCalib][stageIdx]
        X_train, X_test, y_train, y_test = dataset.getStratifiedTrainingSet()
        normalizer = ImageNormalizer(posDatasetFilePath, negDatasetFilePath, model.getNormalizationMethod())
        precisions, recalls, thresholds = precision_recall_curve(y_test, model.predict(X_test, normalizer)[:, 1])
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()