from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import numpy as np

from .model import MODELS

from .data import numDetectionWindowsAlongAxis, squashCoords, MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS_ARR, OBJECT_DATABASE_PATHS, NEGATIVE_DATABASE_PATHS
from .preprocess import ImageNormalizer

IOU_THRESH = .5
NET_12_THRESH = .0032
NET_24_THRESH = .357
NET_48_THRESH = .5

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

MODELS = [(MODELS[False][stageIdx], MODELS[True][stageIdx]) for stageIdx in np.arange(len(SCALES))]
PATHS = []
NORMALIZERS = []
THRESHOLDS = (NET_12_THRESH, NET_24_THRESH, NET_48_THRESH)

for stageIdx in np.arange(1):
    PATHS.append((OBJECT_DATABASE_PATHS[stageIdx], NEGATIVE_DATABASE_PATHS[stageIdx]))
    NORMALIZERS.append(tuple((ImageNormalizer(PATHS[stageIdx][0], PATHS[stageIdx][1], MODELS[stageIdx][isCalib].getNormalizationMethod()) for isCalib in (0, 1))))

def detectMultiscale(img, maxStageIdx=len(SCALES)-1, minFaceScale = MIN_FACE_SCALE):
    for stageIdx in np.arange(0, maxStageIdx + 1):
        curScale = SCALES[stageIdx][0]
        classifierInputs = []
        calibratorInputs = []

        if stageIdx == 0:
            detectionWindowGenerator = getDetectionWindows(img, curScale)
            totalNumDetectionWindows = next(detectionWindowGenerator)
            detectionWindows = np.zeros((totalNumDetectionWindows, curScale, curScale, 3))
            coords = np.zeros((totalNumDetectionWindows, 4))

            for i, (xMin, yMin, xMax, yMax, detectionWindow) in enumerate(detectionWindowGenerator):
                detectionWindows[i] = detectionWindow
                coords[i] = (xMin, yMin, xMax, yMax)

            coords *= minFaceScale/curScale
        else:
            for i in np.arange(0, stageIdx):
                classifierInputs[i] = classifierInputs[i][posDetectionIndices][picked]

        classifierNormalizer, calibNormalizer = NORMALIZERS[stageIdx]
        classifier, calibrator = MODELS[stageIdx]

        detectionWindows = detectionWindows if stageIdx == 0 else getNetworkInputs(img, curScale, coords).astype(np.float)
        classifierInputs.append(classifierNormalizer.preprocess(detectionWindows))
        predictions = classifier.predict(classifierInputs)[:,1]
        posDetectionIndices = np.where(predictions>=THRESHOLDS[stageIdx])

        calibratorInputs = [calibNormalizer.preprocess(detectionWindows[posDetectionIndices])]
        calibPredictions = calibrator.predict(calibratorInputs)
        coords = calibrateCoordinates(coords[posDetectionIndices], calibPredictions)
        coords, picked = nms(coords, predictions[posDetectionIndices])
        
        if stageIdx == maxStageIdx:
            return coords.astype(np.int32, copy=False)
