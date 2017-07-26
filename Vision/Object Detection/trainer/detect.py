from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import numpy as np
import os

from .model import MODELS

from .data import numDetectionWindowsAlongAxis, squashCoords, MIN_OBJECT_SCALE, OFFSET, SCALES, CALIB_PATTERNS_ARR, OBJECT_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS
from .preprocess import ImageNormalizer

IOU_THRESH = .5
NET_12_THRESH = .01
NET_24_THRESH = .06
NET_48_THRESH = .3

NUM_PYRAMID_LEVELS = 5
PYRAMID_DOWNSCALE = 1.25

NUM_CALIBRATION_STEPS = 1

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

def local_nms(boxes, predictions, pyrIdxs, iouThresh = IOU_THRESH):
    suppressedBoxes = np.zeros((0, *boxes.shape[1:]))
    prevIdx = 0
    newPyrIdxs = []
    picked = []

    for curIdx in pyrIdxs:
        localSuppressedBoxes, localPicked = nms(boxes[prevIdx:curIdx], predictions[prevIdx:curIdx], iouThresh)
        suppressedBoxes = np.vstack((suppressedBoxes, localSuppressedBoxes))
        picked.extend(localPicked)
        prevIdx = curIdx
        newPyrIdxs.append(suppressedBoxes.shape[0])

    return suppressedBoxes, picked, newPyrIdxs

def getDetectionWindows(img, scale, minObjectScale = MIN_OBJECT_SCALE, numPyramidLevels = NUM_PYRAMID_LEVELS, pyramidDownscale = PYRAMID_DOWNSCALE):
    numPyramidLevels = min(int(np.log(min(img.shape[:2])/minObjectScale)/np.log(pyramidDownscale)), numPyramidLevels)
    resized = [cv2.resize(img, None, fx = scale/(minObjectScale*pyramidDownscale**i), fy = scale/(minObjectScale*pyramidDownscale**i)) for i in np.arange(numPyramidLevels)]
    numDetectionWindows = sum((numDetectionWindowsAlongAxis(img.shape[0])*numDetectionWindowsAlongAxis(img.shape[1]) for img in resized))
    yield numDetectionWindows

    for pyrLevel, img in enumerate(resized):
        for yIdx in np.arange(numDetectionWindowsAlongAxis(img.shape[0])):
            for xIdx in np.arange(numDetectionWindowsAlongAxis(img.shape[1])):
                xMin, yMin, xMax, yMax = (xIdx*OFFSET, yIdx*OFFSET, xIdx*OFFSET+scale, yIdx*OFFSET+scale)
                yield (pyrLevel, xMin, yMin, xMax, yMax, img[yMin:yMax, xMin:xMax])

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

MODELS = [(MODELS[False][stageIdx], MODELS[True][stageIdx]) for stageIdx in np.arange(len(SCALES))]
PATHS = []
NORMALIZERS = []
THRESHOLDS = (NET_12_THRESH, NET_24_THRESH, NET_48_THRESH)

for stageIdx in np.arange(len(SCALES)):
    if os.path.isfile(OBJECT_DATABASE_PATHS[stageIdx]) and os.path.isfile(NEGATIVE_DATABASE_PATHS[stageIdx]):
        PATHS.append((OBJECT_DATABASE_PATHS[stageIdx], NEGATIVE_DATABASE_PATHS[stageIdx]))
        NORMALIZERS.append(tuple((ImageNormalizer(PATHS[stageIdx][0], PATHS[stageIdx][1], MODELS[stageIdx][isCalib].getNormalizationMethod()) for isCalib in (0, 1))))

def detectMultiscale(img, maxStageIdx=len(SCALES)-1, minObjectScale = MIN_OBJECT_SCALE):
    classifierInputs = []
    calibratorInputs = []

    for stageIdx in np.arange(0, maxStageIdx + 1):
        curScale = SCALES[stageIdx][0]

        if stageIdx == 0:
            detectionWindowGenerator = getDetectionWindows(img, curScale)
            totalNumDetectionWindows = next(detectionWindowGenerator)
            detectionWindows = np.zeros((totalNumDetectionWindows, curScale, curScale, 3))
            coords = np.zeros((totalNumDetectionWindows, 4))
            pyrIdxs = []
            prevPyrLevel = 0

            for i, (pyrLevel, xMin, yMin, xMax, yMax, detectionWindow) in enumerate(detectionWindowGenerator):
                detectionWindows[i] = detectionWindow
                coords[i] = (xMin, yMin, xMax, yMax)
                coords[i] *= PYRAMID_DOWNSCALE ** pyrLevel

                if pyrLevel != prevPyrLevel:
                    prevPyrLevel = pyrLevel
                    pyrIdxs.append(i)
            
            pyrIdxs.append(len(detectionWindows))
            coords *= minObjectScale/curScale
        else:
            for i in np.arange(0, stageIdx):
                classifierInputs[i] = classifierInputs[i][posDetectionIndices][picked]
        
        classifierNormalizer, calibNormalizer = NORMALIZERS[stageIdx]
        classifier, calibrator = MODELS[stageIdx]

        detectionWindows = detectionWindows if stageIdx == 0 else getNetworkInputs(img, curScale, coords).astype(np.float)
        classifierInputs.insert(0 if stageIdx < 2 else stageIdx, classifierNormalizer.preprocess(detectionWindows))
        predictions = classifier.predict(classifierInputs)[:,1]
        posDetectionIndices = np.where(predictions>=THRESHOLDS[stageIdx])

        calibratorInputs = [calibNormalizer.preprocess(detectionWindows[posDetectionIndices])]
        calibPredictions = calibrator.predict(calibratorInputs)
        coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)

        if stageIdx == len(SCALES)-1:
            coords, picked = nms(coords, predictions[posDetectionIndices], iouThresh = .3)
        else:
            coords, picked, pyrIdxs = local_nms(coords, predictions[posDetectionIndices], pyrIdxs)
        
        if stageIdx == maxStageIdx:
            return coords.astype(np.int32, copy=False)

def fastDetect(img, getDetectionWindows, numCalibSteps = NUM_CALIBRATION_STEPS):
    stageIdx = 0
    curScale = SCALES[stageIdx][0]

    objectProposals, centers = getDetectionWindows(img)
    centers = np.asarray(centers)
    totalNumDetectionWindows = len(objectProposals)
    detectionWindows = np.zeros((totalNumDetectionWindows, curScale, curScale, 3))
    coords = np.zeros((totalNumDetectionWindows, 4))

    for i, (xMin, yMin, xMax, yMax) in enumerate(objectProposals):
        xMin, yMin, w, h = squashCoords(img, xMin, yMin, xMax-xMin, yMax-yMin)
        coords[i] = (xMin, yMin, xMin + w, yMin + h)
        detectionWindows[i] = cv2.resize(img[yMin:yMin+h, xMin:xMin+w], (curScale, curScale))

    classifierNormalizer, calibNormalizer = NORMALIZERS[stageIdx]
    classifier, calibrator = MODELS[stageIdx]

    predictions = classifier.predict([classifierNormalizer.preprocess(detectionWindows)])[:,1]
    posDetectionIndices = np.where(predictions>=THRESHOLDS[stageIdx])
    coords, centers, predictions, detectionWindows = coords[posDetectionIndices], centers[posDetectionIndices], predictions[posDetectionIndices], detectionWindows[posDetectionIndices]

    for i in np.arange(numCalibSteps-1):
        detectionWindows = getNetworkInputs(img, curScale, coords).astype(np.float)
        predictions = classifier.predict([classifierNormalizer.preprocess(detectionWindows)])[:,1]
        
        calibPredictions = calibrator.predict([calibNormalizer.preprocess(detectionWindows)])
        coords = calibrateCoordinates(coords, calibPredictions)

        coords, picked = nms(coords, predictions)
        centers = centers[picked]

    return coords.astype(np.int32, copy=False), centers