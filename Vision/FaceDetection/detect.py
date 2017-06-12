#!/usr/bin/env python3.5
import math

import cv2
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

session = tf.Session()
K.set_session(session)

from data import numDetectionWindowsAlongAxis, squashCoords, MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS_ARR
from util import static_vars

NET_FILE_NAMES = {False: {SCALES[0][0]: '12net.hdf', SCALES[1][0]: '24net.hdf'}, 
                  True: {SCALES[0][0]: '12calibnet.hdf', SCALES[1][0]: '24calibnet.hdf'}}
IOU_THRESH = .5
PYRAMID_DOWNSCALE = 2
NET_12_THRESH = .05
NET_24_THRESH = .05

def to_tf_model(func):
    from train import INPUT_LAYER_NAME_ID

    def decorate(*args, **kwargs):
        inputLayers = []
        ret = func(*args, **kwargs)

        for layer in ret.layers:
            if layer.name.startswith(INPUT_LAYER_NAME_ID):
                inputLayers.append(layer.input)

        inputLayers.append(K.learning_phase())

        return K.function(inputLayers, [ret.layers[-1].output])

    return decorate

@to_tf_model
def load_12net():
    return load_model(NET_FILE_NAMES[False].get(SCALES[0][0]))

@to_tf_model
def load_12netcalib():
    return load_model(NET_FILE_NAMES[True].get(SCALES[0][0]))

@to_tf_model
def load_24net():
    return load_model(NET_FILE_NAMES[False].get(SCALES[1][0]))

@to_tf_model
def load_24netcalib():
    return load_model(NET_FILE_NAMES[True].get(SCALES[1][0]))


def IoU(boxes, box, area=None):
    int_x1, int_y1, int_x2, int_y2 = ((np.maximum if i < 2 else np.minimum)(boxes[:, i], box[i]) for i in np.arange(boxes.shape[1]))
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,0]) if not area else area
    int_area = (np.maximum(0,int_x2-int_x1))*(np.maximum(0,int_y2-int_y1))
    union_area = area+(box[2]-box[0]+1)*(box[3]-box[1]+1)-int_area
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

def local_nms(boxes, predictions, pyrIdxs, iouThresh = IOU_THRESH):
    suppressedBoxes = np.zeros((0, *boxes.shape[1:]))
    prevIdx = 0
    newIdxs = []
    picked = []

    for curIdx in pyrIdxs:
        localSuppressedBoxes, localPicked = nms(boxes[prevIdx:curIdx], predictions[prevIdx:curIdx], iouThresh)
        picked.extend(localPicked)
        suppressedBoxes = np.vstack((suppressedBoxes, localSuppressedBoxes))
        prevIdx = curIdx
        newIdxs.append(suppressedBoxes.shape[0])

    return suppressedBoxes, newIdxs, picked

def getImagePyramid(img, minSize, downscale = PYRAMID_DOWNSCALE):
    imgs = []

    while img.shape[1] >= minSize[0] and img.shape[0] >= minSize[1]:
        imgs.append(img)
        img = cv2.resize(img, None, fx=1/downscale, fy=1/downscale)

    return imgs

def getDetectionWindows(img, scale, pyrDownscale = PYRAMID_DOWNSCALE, minFaceScale = MIN_FACE_SCALE):
    imgPyr = getImagePyramid(img, (minFaceScale, minFaceScale), PYRAMID_DOWNSCALE)
    resized = [cv2.resize(pyrImg, None, fx = scale/minFaceScale, fy = scale/minFaceScale) for pyrImg in imgPyr]
    numDetectionWindows = sum((numDetectionWindowsAlongAxis(resizedImg.shape[0])*numDetectionWindowsAlongAxis(resizedImg.shape[1]) for resizedImg in resized))
    yield numDetectionWindows

    for pyrLevel, resizedImg, in enumerate(resized):
        for yIdx in np.arange(numDetectionWindowsAlongAxis(resizedImg.shape[0])):
            for xIdx in np.arange(numDetectionWindowsAlongAxis(resizedImg.shape[1])):
                xMin, yMin, xMax, yMax = (xIdx*OFFSET, yIdx*OFFSET, xIdx*OFFSET+scale, yIdx*OFFSET+scale)
                yield (pyrLevel, xMin, yMin, xMax, yMax, resizedImg[yMin:yMax, xMin:xMax])

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
@static_vars(classifiers=_createModelDict(), calibrators=_createModelDict())
def detectMultiscale(img, maxStageIdx=len(SCALES)-1, minFaceScale = MIN_FACE_SCALE):
    from train import preprocessImages
    from FaceDetection import PROFILE
    curScale = SCALES[0][0]

    if detectMultiscale.classifiers.get(curScale) is None:
        detectMultiscale.classifiers[curScale] = load_12net()
        detectMultiscale.calibrators[curScale] = load_12netcalib()

    detectionWindowGenerator = getDetectionWindows(img, curScale, PYRAMID_DOWNSCALE)
    totalNumDetectionWindows = next(detectionWindowGenerator)
    detectionWindows = np.zeros((totalNumDetectionWindows, curScale, curScale, 3))
    coords = np.zeros((totalNumDetectionWindows, 4))
    pyrIdxs = []
    prevPyrLevel = None

    for i, (pyrLevel, xMin, yMin, xMax, yMax, detectionWindow) in enumerate(detectionWindowGenerator):
        detectionWindows[i] = detectionWindow
        coords[i] = (xMin, yMin, xMax, yMax)
        coords[i] *= PYRAMID_DOWNSCALE**pyrLevel*(minFaceScale/curScale)
        
        if pyrLevel != prevPyrLevel and pyrLevel != 0:
            pyrIdxs.append(i)
            prevPyrLevel = pyrLevel

    detectionWindows = preprocessImages(detectionWindows.astype(np.float))
    predictions = detectMultiscale.classifiers[curScale]([detectionWindows, 0])[0][:,1]
    posDetectionIndices = np.where(predictions>=NET_12_THRESH)

    detectionWindows = detectionWindows[posDetectionIndices]
    calibPredictions = detectMultiscale.calibrators[curScale]([detectionWindows, 0])[0]
    coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)
    coords, pyrIdxs, picked = local_nms(coords, predictions[posDetectionIndices], pyrIdxs)

    if maxStageIdx == 0:
        return coords.astype(np.int32, copy=False)

    curScale = SCALES[1][0]

    if detectMultiscale.classifiers.get(curScale) is None:
        detectMultiscale.classifiers[curScale] = load_24net()
        detectMultiscale.calibrators[curScale] = load_24netcalib()

    net24_inputs = preprocessImages(getNetworkInputs(img, curScale, coords).astype(np.float))
    predictions = detectMultiscale.classifiers[curScale]([detectionWindows[picked], net24_inputs, 0])[0][:, 1]
    posDetectionIndices = np.where(predictions>=NET_24_THRESH)

    net24_inputs = net24_inputs[posDetectionIndices]
    calibPredictions = detectMultiscale.calibrators[curScale]([net24_inputs, 0])[0]
    coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)
    coords, pyrIdxs, picked = local_nms(coords, predictions[posDetectionIndices], pyrIdxs)
    return coords.astype(np.int32, copy=False)

