import cv2
import numpy as np

import keras
from keras.models import load_model
from keras.optimizers import Adam
from train import build12net, build12calibNet

from data import MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS
from annotation import RectangleAnnotation

TWELVE_NET_FILE_NAME = '12net.hdf'
TWELVE_CALIB_NET_FILE_NAME = '12calibnet.hdf'

OVERLAP_THRESH = .4

def numDetectionWindowsAlongAxis(size):
    return (size-12)//OFFSET+1

def load_12net():
    classifier = build12net()
    classifier.load_weights(TWELVE_NET_FILE_NAME)
    return classifier

def load_12netcalib():
    calibrator = build12calibNet()
    calibrator.load_weights(TWELVE_CALIB_NET_FILE_NAME)
    return calibrator

classifier = None
calibrator = None

# Malisiewicz et al. (http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
def nms(detections, overlapThresh):
    boxes = np.zeros((len(detections), 4))
    
    for i, detection in enumerate(detections):
        boxes[i] = np.concatenate([detection.top_left, detection.bottom_right])
    
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap >= overlapThresh)[0])))

    return [detections[idx] for idx in pick]

def stage1_predict(mat):
    global classifier
    global calibrator

    if classifier is None:
        classifier = load_12net()
        calibrator = load_12netcalib()

    scale = SCALES[0][0]
    resized = cv2.resize(mat, None, fx = scale/MIN_FACE_SCALE, fy = scale/MIN_FACE_SCALE)
    posDetections = []
    numDetectionWindows = numDetectionWindowsAlongAxis(resized.shape[0])*numDetectionWindowsAlongAxis(resized.shape[1])
    rois = np.ones((numDetectionWindows, scale, scale, 3), dtype = resized.dtype)
    coords = np.ones((numDetectionWindows, 2), dtype=np.int)
    i = 0

    for yIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[0])):
        for xIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[1])):
            coords[i] = np.array([xIdx, yIdx], dtype = np.int)*OFFSET
            top_left = coords[i]
            rois[i] = resized[top_left[1]:top_left[1]+scale,top_left[0]:top_left[0]+scale]
            i += 1
    
    posDetectionIndices = np.where(classifier.predict(rois)[:,1]>=.5)
    numDetections = posDetectionIndices[0].shape[0]
    detections = np.ones((numDetections, scale, scale, 3), resized.dtype)
    coords *= MIN_FACE_SCALE//scale

    for j, idx in enumerate(posDetectionIndices[0]):
        center = coords[idx]+MIN_FACE_SCALE//2
        posDetections.append(RectangleAnnotation(MIN_FACE_SCALE, MIN_FACE_SCALE, *center))
        cropped = posDetections[-1].cropOut(mat, scale, scale)
        detections[j] = cropped

    calibPredictions = np.argmax(calibrator.predict(detections), 1)
    for j, calibIdx in enumerate(calibPredictions):
        posDetections[j].applyTransform(*CALIB_PATTERNS[calibIdx])

    return nms(posDetections,OVERLAP_THRESH)