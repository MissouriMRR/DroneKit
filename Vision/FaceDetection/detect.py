import math
import cv2
import numpy as np

import keras
from keras.models import load_model
from keras.engine.topology import InputLayer
from keras import backend as K

from data import numDetectionWindowsAlongAxis, squashCoords, MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS_ARR
from util import static_vars

NET_FILE_NAMES = {False: {SCALES[0][0]: '12net.hdf', SCALES[1][0]: '24net.hdf', SCALES[2][0]: '48net.hdf'}, 
                  True: {SCALES[0][0]: '12calibnet.hdf', SCALES[1][0]: '24calibnet.hdf', SCALES[2][0]: '48calibnet.hdf'}}
IOU_THRESH = .5
NET_12_THRESH = .005
NET_24_THRESH = .219
NET_48_THRESH = .5

def to_tf_model(func):
    def decorate(*args, **kwargs):
        inputLayers = []
        ret = func(*args, **kwargs)

        for layer in ret.layers:
            if type(layer) is InputLayer:
                inputLayers.append(layer.input)

        inputLayers.append(K.learning_phase())

        return K.function(inputLayers, [ret.layers[-1].output])

    return decorate

@to_tf_model
def loadNet(stageIdx, isCalib):
    return MODELS[isCalib][stageIdx].loadModel()

def getNormalizationMethod(stageIdx, isCalib):
    return MODELS[isCalib][stageIdx].getNormalizationMethod()

def IoU(boxes, box, area=None):
    int_x1, int_y1, int_x2, int_y2 = ((np.maximum if i < 2 else np.minimum)(boxes[:, i], box[i]) for i in np.arange(boxes.shape[1]))
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:, 1]) if not area else area
    int_area = (np.maximum(0,int_x2-int_x1))*(np.maximum(0,int_y2-int_y1))
    union_area = area+(box[2]-box[0])*(box[3]-box[1])-int_area
    int_over_union = int_area/union_area
    return int_over_union

def nms(boxes, predictions, iouThresh = IOU_THRESH):
    idxs = np.argsort(predictions)
    picked = []

    while len(idxs) > 0:
        pick = boxes[idxs[-1]]
        picked.append(idxs[-1])

        int_over_union = IoU(boxes[idxs[:-1]], pick)
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1],np.where(int_over_union > iouThresh)[0])))
    
    return boxes[picked], picked

def getDetectionWindows(img, scale, minFaceScale = MIN_FACE_SCALE):
    resized = cv2.resize(img, None, fx = scale/minFaceScale, fy = scale/minFaceScale)
    numDetectionWindows = numDetectionWindowsAlongAxis(resized.shape[0])*numDetectionWindowsAlongAxis(resized.shape[1])
    yield numDetectionWindows

    for yIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[0])):
        for xIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[1])):
            xMin, yMin, xMax, yMax = (xIdx*OFFSET, yIdx*OFFSET, xIdx*OFFSET+scale, yIdx*OFFSET+scale)
            yield (xMin, yMin, xMax, yMax, resized[yMin:yMax, xMin:xMax])

def calibrateCoordinates(coords, calibPredictions):
    calibTransformations = CALIB_PATTERNS_ARR[np.argmax(calibPredictions, axis=1)]
    sn = calibTransformations[:, 0]
    dimensions = (coords[:, 2:]-coords[:, :2])/sn[:, None]
    coords[:, :2] -= dimensions*calibTransformations[:, 1:]
    coords[:, 2:] = coords[:, :2] + dimensions
    return coords

def getNetworkInputs(img, curScale, coords):
    inputs = np.ones((len(coords), curScale, curScale, 3), dtype=np.uint8)

    for i, (xMin, yMin, xMax, yMax) in enumerate(coords.astype(np.int32)):
        xMin, yMin, w, h = squashCoords(img, xMin, yMin, xMax-xMin, yMax-yMin)
        inputs[i] = cv2.resize(img[yMin:yMin+h, xMin:xMin+w], (curScale, curScale))

    return inputs

_createModelDict = lambda: {SCALES[i][0]:None for i in range(len(SCALES))}
@static_vars(classifiers=_createModelDict(), calibrators=_createModelDict(), preprocessors={})
def detectMultiscale(img, maxStageIdx=len(SCALES)-1, minFaceScale = MIN_FACE_SCALE):
    from model import MODELS
    from FaceDetection import PROFILE
    from data import FACE_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS
    from preprocess import ImageNormalizer

    curScale = SCALES[0][0]
    calibNormalizer = None
    classifierNormalizer = None
    posDatasetPath, negDatasetPath = (FACE_DATABASE_PATHS[0], NEGATIVE_DATABASE_PATHS[0])

    if detectMultiscale.classifiers.get(curScale) is None:
        detectMultiscale.classifiers[curScale], detectMultiscale.calibrators[curScale] = (loadNet(0, isCalib) for isCalib in (False, True))
        detectMultiscale.preprocessors[curScale] = {norm: ImageNormalizer(posDatasetPath, negDatasetPath, norm) for norm in ImageNormalizer.NORM_METHODS}

    calibNormalizer, classifierNormalizer = (detectMultiscale.preprocessors[curScale][getNormalizationMethod(0, isCalib)] for isCalib in (False, True))
        
    detectionWindowGenerator = getDetectionWindows(img, curScale)
    totalNumDetectionWindows = next(detectionWindowGenerator)
    detectionWindows = np.zeros((totalNumDetectionWindows, curScale, curScale, 3))
    coords = np.zeros((totalNumDetectionWindows, 4))

    for i, (xMin, yMin, xMax, yMax, detectionWindow) in enumerate(detectionWindowGenerator):
        detectionWindows[i] = detectionWindow
        coords[i] = (xMin, yMin, xMax, yMax)

    coords *= minFaceScale/curScale

    predictions = detectMultiscale.classifiers[curScale]([classifierNormalizer.preprocess(detectionWindows), 0])[0][:,1]
    posDetectionIndices = np.where(predictions>=NET_12_THRESH)

    detectionWindows = detectionWindows[posDetectionIndices]
    calibPredictions = detectMultiscale.calibrators[curScale]([calibNormalizer.preprocess(detectionWindows), 0])[0]
    coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)
    coords, picked = nms(coords, predictions[posDetectionIndices])

    if maxStageIdx == 0:
        return coords.astype(np.int32, copy=False)

    curScale = SCALES[1][0]
    posDatasetPath, negDatasetPath = (FACE_DATABASE_PATHS[1], NEGATIVE_DATABASE_PATHS[1])

    if detectMultiscale.classifiers.get(curScale) is None:
        detectMultiscale.classifiers[curScale], detectMultiscale.calibrators[curScale] = (loadNet(1, isCalib) for isCalib in (False, True))
        detectMultiscale.preprocessors[curScale] = {norm: ImageNormalizer(posDatasetPath, negDatasetPath, norm) for norm in ImageNormalizer.NORM_METHODS}

    detectionWindows = classifierNormalizer.preprocess(detectionWindows[picked])
    calibNormalizer, classifierNormalizer = (detectMultiscale.preprocessors[curScale][getNormalizationMethod(1, isCalib)] for isCalib in (False, True))

    net24_inputs = getNetworkInputs(img, curScale, coords).astype(np.float)
    predictions = detectMultiscale.classifiers[curScale]([classifierNormalizer.preprocess(net24_inputs), detectionWindows, 0])[0][:, 1]
    posDetectionIndices = np.where(predictions>=NET_24_THRESH)

    net24_inputs = net24_inputs[posDetectionIndices]
    calibPredictions = detectMultiscale.calibrators[curScale]([calibNormalizer.preprocess(net24_inputs), 0])[0]
    coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)
    coords, picked= nms(coords, predictions[posDetectionIndices])

    if maxStageIdx == 1:
        return coords.astype(np.int32, copy=False)

    curScale = SCALES[2][0]

    if detectMultiscale.classifiers.get(curScale) is None:
        detectMultiscale.classifiers[curScale] = load_48net()
        detectMultiscale.calibrators[curScale] = load_48netcalib()

    net48_inputs = preprocessImages(getNetworkInputs(img, curScale, coords).astype(np.float))
    predictions = detectMultiscale.classifiers[curScale]([net24_inputs[picked], net48_inputs, 0])[0][:, 1]
    posDetectionIndices = np.where(predictions>=NET_48_THRESH)

    calibPredictions = detectMultiscale.calibrators[curScale]([net48_inputs[posDetectionIndices], 0])[0]
    coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)
    coords, picked = nms(coords, predictions[posDetectionIndices], .5)
    return coords.astype(np.int32, copy=False)
