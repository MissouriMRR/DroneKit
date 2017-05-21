import cv2
import numpy as np

import keras
from keras.models import load_model
from keras.optimizers import Adam
from train import build12net

from data import MIN_FACE_SCALE, OFFSET, SCALES
from annotation import RectangleAnnotation

TWELVE_NET_FILE_NAME = '12net.hdf'
TWELVE_CALIB_NET_FILE_NAME = '12calibnet.hdf'

def numDetectionWindowsAlongAxis(size):
    return (size-12)//OFFSET+1

def load_12net():
    classifier = build12net()
    classifier.load_weights(TWELVE_NET_FILE_NAME)
    return classifier

classifier = None


def stage1_predict(mat):
    global classifier

    if classifier is None:
        classifier = load_12net()

    scale = SCALES[0][0]
    resized = cv2.resize(mat, None, fx = scale/MIN_FACE_SCALE, fy = scale/MIN_FACE_SCALE)
    posDetections = []
    numDetectionWindows = numDetectionWindowsAlongAxis(resized.shape[0])*numDetectionWindowsAlongAxis(resized.shape[1])
    rois = np.ones((numDetectionWindows, 12, 12, 3), dtype = resized.dtype)
    coords = np.ones((numDetectionWindows, 2), dtype=np.int)
    i = 0

    for yIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[0])):
        for xIdx in np.arange(numDetectionWindowsAlongAxis(resized.shape[1])):
            coords[i] = np.array([xIdx, yIdx], dtype = np.int)*OFFSET
            top_left = coords[i]
            rois[i] = resized[top_left[1]:top_left[1]+scale,top_left[0]:top_left[0]+scale]
            i += 1
    
    posDetectionIndices = np.where(classifier.predict(rois)[:,1]>=.5)
    coords *= MIN_FACE_SCALE//scale


    for idx in posDetectionIndices[0]:
        center = coords[idx]+MIN_FACE_SCALE//2
        posDetections.insert(-1, RectangleAnnotation(MIN_FACE_SCALE, MIN_FACE_SCALE, *center))

    return posDetections