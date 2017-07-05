import pickle
import h5py
import os

from sklearn.metrics import f1_score

from dataset import ClassifierDataset
from preprocess import ImageNormalizer
from model import DEFAULT_NUM_EPOCHS

def train(stageIdx, trainCalib, numEpochs = DEFAULT_NUM_EPOCHS, tune = True):
    from model import MODELS
    from data import FACE_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS, CALIBRATION_DATABASE_PATHS, SCALES, DATASET_LABEL, LABELS_LABEL, \
                     createFaceDataset, createNegativeDataset, createCalibrationDataset, mineNegatives

    model = MODELS[trainCalib][stageIdx]
    posDatasetFilePath = FACE_DATABASE_PATHS[stageIdx]
    negDatasetFilePath = NEGATIVE_DATABASE_PATHS[stageIdx]
    calibDatasetFilePath = CALIBRATION_DATABASE_PATHS[SCALES[stageIdx][0]]
    datasetCreationMethod = {posDatasetFilePath: createFaceDataset, 
                             negDatasetFilePath: createNegativeDataset if stageIdx == 0 else mineNegatives, 
                             calibDatasetFilePath: createCalibrationDataset}
    labels = None
    paths = [posDatasetFilePath, negDatasetFilePath]

    for filePath in datasetCreationMethod.keys():
        if not os.path.isfile(filePath) and (filePath != calibDatasetFilePath or trainCalib):
            datasetCreationMethod[filePath](stageIdx)

    if trainCalib:
        with h5py.File(calibDatasetFilePath, 'r') as calibDatasetFile:
            labels = calibDatasetFile[LABELS_LABEL][:]
            paths = [calibDatasetFilePath, None]

    if not model.wasTuned() and tune:
        model.tune(posDatasetFilePath, negDatasetFilePath, paths, labels, f1_score)

    with ClassifierDataset(*paths, labels) as dataset:
        X_train, X_test, y_train, y_test = dataset.getStratifiedTrainingSet()
        normalizer = ImageNormalizer(posDatasetFilePath, negDatasetFilePath, model.getNormalizationMethod())
        normalizer.addDataAugmentationParams(model.getNormalizationParams())
        model.fit(X_train, X_test, y_train, y_test, normalizer, numEpochs)