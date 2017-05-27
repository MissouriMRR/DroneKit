import cv2
import numpy as np

import keras
from keras.models import load_model
from keras.optimizers import Adam
from train import build12net, build12calibNet

from data import MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS
from annotation import RectangleAnnotation
from util import static_vars, annotations2matrix, detections2boxes

TWELVE_NET_FILE_NAME = '12net.hdf'
TWELVE_CALIB_NET_FILE_NAME = '12calibnet.hdf'

IOU_THRESH = .5

def numDetectionWindowsAlongAxis(size):
    return (size-12)//OFFSET+1

def load_12net():
    return load_model(TWELVE_NET_FILE_NAME)

def load_12netcalib():
    return load_model(TWELVE_CALIB_NET_FILE_NAME)

@annotations2matrix
def IoU(boxes, box, area=None):
    int_x1, int_y1, int_x2, int_y2 = ((np.maximum if i < 2 else np.minimum)(boxes[:, i], box[i]) for i in np.arange(boxes.shape[1]))
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,0]) if not area else area
    int_area = (np.maximum(0,int_x2-int_x1))*(np.maximum(0,int_y2-int_y1))
    union_area = area+(box[2]-box[0])*(box[3]-box[1])-int_area
    int_over_union = int_area/union_area
    return int_over_union

def nms(detections, iouThresh, predictions = None):
    boxes = detections2boxes(detections)
    idxs = np.argsort(predictions) if predictions is not None else np.arange(boxes.shape[0])
    picked = []

    while len(idxs) > 0:
        pick = boxes[idxs[-1]]
        picked.append(idxs[-1])

        int_over_union = IoU(boxes[idxs[:-1]], pick)
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1],np.where(int_over_union > iouThresh)[0])))
    
    return [detections[idx] for idx in picked]

@static_vars(classifier = None, calibrator = None)
def stage1_predict(mat, iouThresh = IOU_THRESH, minFaceScale = MIN_FACE_SCALE):
    if stage1_predict.classifier is None:
        stage1_predict.classifier = load_12net()
    if stage1_predict.calibrator is None:
        stage1_predict.calibrator = load_12netcalib()

    scale = SCALES[0][0]
    resized = cv2.resize(mat, None, fx = scale/minFaceScale, fy = scale/minFaceScale)
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
    
    predictions = stage1_predict.classifier.predict(rois/255)[:,1]
    posDetectionIndices = np.where(predictions>=.5)
    numDetections = posDetectionIndices[0].shape[0]
    detections = np.ones((numDetections, scale, scale, 3), resized.dtype)
    coords *= minFaceScale//scale

    for j, idx in enumerate(posDetectionIndices[0]):
        center = coords[idx]+minFaceScale//2
        posDetections.append(RectangleAnnotation(minFaceScale, minFaceScale, *center))
        cropped = posDetections[-1].cropOut(mat, scale, scale)
        detections[j] = cropped

    if len(detections) > 0:
        calibPredictions = np.argmax(stage1_predict.calibrator.predict(detections), 1)
        for j, calibIdx in enumerate(calibPredictions):
            posDetections[j].applyTransform(*CALIB_PATTERNS[calibIdx])

    return nms(posDetections, iouThresh, predictions[posDetectionIndices])

def stage1_predict_multiscale(mat, iouThresh = IOU_THRESH, SCALES = np.arange(MIN_FACE_SCALE, MIN_FACE_SCALE*3, MIN_FACE_SCALE//2)):
    detections = []

    for scale in SCALES:
        detections.extend(stage1_predict(mat, iouThresh, scale))

    return nms(detections, iouThresh)