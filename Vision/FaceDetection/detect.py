#!/usr/bin/env python3.5
import cv2
import numpy as np

import keras
from keras.models import load_model

from data import numDetectionWindowsAlongAxis, MIN_FACE_SCALE, OFFSET, SCALES, CALIB_PATTERNS
from util import static_vars

NET_FILE_NAMES = {False: {SCALES[0][0]: '12net.hdf', SCALES[1][0]: '24net.hdf'}, 
                  True: {SCALES[0][0]: '12calibnet.hdf', SCALES[1][0]: '24calibnet.hdf'}}
IOU_THRESH = .5

def load_12net():
    return load_model(NET_FILE_NAMES.get(SCALES[0][0]))

def load_12netcalib():
    return load_model(CALIB_NET_FILE_NAMES.get(SCALES[0][0]))

def load_24net():
    return load_model(NET_FILE_NAMES.get(SCALES[1][0]))

def load_24netcalib():
    return load_model(CALIB_NET_FILE_NAMES.get(SCALES[1][0]))


def IoU(boxes, box, area=None):
    int_x1, int_y1, int_x2, int_y2 = ((np.maximum if i < 2 else np.minimum)(boxes[:, i], box[i]) for i in np.arange(boxes.shape[1]))
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,0]) if not area else area
    int_area = (np.maximum(0,int_x2-int_x1))*(np.maximum(0,int_y2-int_y1))
    union_area = area+(box[2]-box[0])*(box[3]-box[1])-int_area
    int_over_union = int_area/union_area
    return int_over_union

