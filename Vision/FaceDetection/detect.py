#!/usr/bin/env python3.5
import math

import cv2
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from skimage.transform import pyramid_gaussian

session = tf.Session()
K.set_session(session)

from data import numDetectionWindowsAlongAxis, MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS_ARR
from util import static_vars

NET_FILE_NAMES = {False: {SCALES[0][0]: '12net.hdf', SCALES[1][0]: '24net.hdf'}, 
                  True: {SCALES[0][0]: '12calibnet.hdf', SCALES[1][0]: '24calibnet.hdf'}}
IOU_THRESH = .5
PYRAMID_DOWNSCALE = 2

def to_tf_model(func):
    def decorate(*args, **kwargs):
        ret = func(*args, **kwargs)
        return K.function([ret.layers[0].input, K.learning_phase()], [ret.layers[-1].output])

    return decorate

@to_tf_model
def load_12net():
    return load_model(NET_FILE_NAMES[False].get(SCALES[0][0]))

@to_tf_model
def load_12netcalib():
    return load_model(NET_FILE_NAMES[True].get(SCALES[0][0]))

def load_24net():
    return load_model(NET_FILE_NAMES[False].get(SCALES[1][0]))

def load_24netcalib():
    return load_model(NET_FILE_NAMES[True].get(SCALES[1][0]))


def IoU(boxes, box, area=None):
    int_x1, int_y1, int_x2, int_y2 = ((np.maximum if i < 2 else np.minimum)(boxes[:, i], box[i]) for i in np.arange(boxes.shape[1]))
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,0]) if not area else area
    int_area = (np.maximum(0,int_x2-int_x1))*(np.maximum(0,int_y2-int_y1))
    union_area = area+(box[2]-box[0])*(box[3]-box[1])-int_area
    int_over_union = int_area/union_area
    return int_over_union

def getDetectionWindows(img, scale, minFaceScale = MIN_FACE_SCALE):
    resized = cv2.resize(img, None, fx = scale/minFaceScale, fy = scale/minFaceScale)
    numDetectionWindows = numDetectionWindowsAlongAxis(resized.shape[0])*numDetectionWindowsAlongAxis(resized.shape[1])
    detectionWindows = np.ones((numDetectionWindows, scale, scale, 3), dtype = np.uint8)
    coords = np.ones((numDetectionWindows, 4))
    i = 0

    for yIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[0])):
        for xIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[1])):
            xMin, yMin, xMax, yMax = (xIdx*OFFSET, yIdx*OFFSET, xIdx*OFFSET+scale, yIdx*OFFSET+scale)
            coords[i] = (xMin, yMin, xMax, yMax)
            detectionWindows[i] = resized[yMin:yMax, xMin:xMax]
            i += 1

    return (detectionWindows, coords*minFaceScale//scale)

def calibrateCoordinates(coords, calibPredictions):
    calibTransformations = CALIB_PATTERNS_ARR[np.argmax(calibPredictions, axis=1)]
    sn = calibTransformations[:, 0]
    dimensions = (coords[:, 2:]-coords[:, :2])/sn[:, None]
    coords[:, :2] -= dimensions*calibTransformations[:, 1:]
    coords[:, 2:] = coords[:, :2] + dimensions
    return coords


_createModelDict = lambda: {SCALES[i][0]:None for i in range(len(SCALES))}
@static_vars(classifiers=_createModelDict(), calibrators=_createModelDict())
def detectMultiscale(img, minFaceScale = MIN_FACE_SCALE):
    from train import preprocessImages
    curScale = SCALES[0][0]

    if detectMultiscale.classifiers.get(curScale) is None:
        detectMultiscale.classifiers[curScale] = load_12net()
        detectMultiscale.calibrators[curScale] = load_12netcalib()


    numPyramidLevels = min(*(math.floor(math.log(img.shape[i]/minFaceScale, PYRAMID_DOWNSCALE)) for i in range(2)))
    detectionWindows, coords = getDetectionWindows(img, curScale)

    for i, img in enumerate(pyramid_gaussian(img, numPyramidLevels, downscale = PYRAMID_DOWNSCALE)):
        newDetectionWindows, newCoords = getDetectionWindows(img, curScale)
        detectionWindows = np.vstack((detectionWindows, newDetectionWindows))
        coords = np.vstack((coords, newCoords * PYRAMID_DOWNSCALE**i))

    detectionWindows = preprocessImages(detectionWindows)
    predictions = detectMultiscale.classifiers[curScale]([detectionWindows, 0])[0]
    posDetectionIndices = np.where(predictions[:,1]>=.3)

    calibPredictions = detectMultiscale.calibrators[curScale]([detectionWindows[posDetectionIndices], 0])[0]
    coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)

    return coords.astype(np.int32, copy=False)
