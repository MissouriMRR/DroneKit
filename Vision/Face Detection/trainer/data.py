from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import random
import os
import math

import h5py
import cv2
import numpy as np
import sqlite3

from .util import static_vars
from .task import DEBUG
from .google_storage import downloadIfAvailable

POSITIVE_IMAGE_DATABASE_FOLDER = 'aflw/data/'
POSITIVE_IMAGE_FOLDER = POSITIVE_IMAGE_DATABASE_FOLDER + 'flickr/'
POSITIVE_IMAGE_DATABASE_FILE = os.path.join(POSITIVE_IMAGE_DATABASE_FOLDER, 'aflw.sqlite')
OBJECT_DATABASE_PATHS = ('data/pos12.hdf', 'data/pos24.hdf', 'data/pos48.hdf')
TEST_IMAGE_DATABASE_FOLDER = 'Annotated Faces in the Wild/FDDB-folds/'
TEST_IMAGES_FOLDER = 'Annotated Faces in the Wild/originalPics'

DATASET_LABEL = 'data'
LABELS_LABEL = 'labels'
CHUNK_SIZE = 256

NEGATIVE_IMAGE_FOLDER = 'Negative Images/images/'
NEGATIVE_DATABASE_PATHS = ('data/neg12.hdf', 'data/neg24.hdf', 'data/neg48.hdf')
TARGET_NUM_NEGATIVES_PER_IMG = 40
TARGET_NUM_NEGATIVES = 300000
MIN_FACE_SCALE = 80
OFFSET = 4

SCALES = ((12,12), (24,24), (48,48))

CALIBRATION_DATABASE_PATHS = {SCALES[0][0]:'data/calib12.hdf', SCALES[1][0]:'data/calib24.hdf', SCALES[2][0]:'data/calib48.hdf'}
TARGET_NUM_CALIBRATION_SAMPLES = 337500
SN = (.83, .91, 1, 1.1, 1.21)
XN = (-.17, 0, .17)
YN = XN
CALIB_PATTERNS = [(sn, xn, yn) for sn in SN for xn in XN for yn in YN]
CALIB_PATTERNS_ARR = np.asarray(CALIB_PATTERNS)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def numDetectionWindowsAlongAxis(size, stageIdx = 0):
    return (size-SCALES[stageIdx][0])//OFFSET+1

@static_vars(faces = [])
def getFaceAnnotations(dbPath = POSITIVE_IMAGE_DATABASE_FILE, posImgFolder = POSITIVE_IMAGE_FOLDER):
    if len(getFaceAnnotations.faces) == 0:
        with sqlite3.connect(dbPath) as conn:
            c = conn.cursor()

            select_string = "faceimages.filepath, facerect.x, facerect.y, facerect.w, facerect.h"
            from_string = "faceimages, faces, facepose, facerect"
            where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
            query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

            for row in c.execute(query_string):
                imgPath = os.path.join(posImgFolder, str(row[0]))

                if os.path.isfile(imgPath):
                    getFaceAnnotations.faces.append((imgPath,) + (row[1:]))

    return getFaceAnnotations.faces

def squashCoords(img, x, y, w, h):
    y = min(max(0, y), img.shape[0])
    x = min(max(0, x), img.shape[1])
    h = min(img.shape[0]-y, h)
    w = min(img.shape[1]-x, w)
    return (x, y, w, h)

def createPositiveDataset(stageIdx, getObjectAnnotations = getFaceAnnotations, posImgFolder = POSITIVE_IMAGE_FOLDER):
    fileName = OBJECT_DATABASE_PATHS[stageIdx]
    resizeTo = SCALES[stageIdx]
    objectAnnotations = getObjectAnnotations(posImgFolder = posImgFolder)
    images = np.zeros((len(objectAnnotations), resizeTo[1], resizeTo[0], 3), dtype = np.uint8)
    curImg = None
    prevImgPath = None

    for i, (imgPath, x, y, w, h) in enumerate(objectAnnotations):
        if imgPath != prevImgPath:
            curImg = cv2.imread(imgPath)

        x, y, w, h = squashCoords(curImg, x, y, w, h)
        images[i] = cv2.resize(curImg[y:y+h,x:x+w], resizeTo)
        prevImgPath = imgPath

    with h5py.File(fileName, 'w') as out:
        out.create_dataset(DATASET_LABEL, data = images, chunks = (CHUNK_SIZE,) + (images.shape[1:]))

def createNegativeDataset(stageIdx, negImgFolder = NEGATIVE_IMAGE_FOLDER, numNegatives = TARGET_NUM_NEGATIVES, numNegativesPerImg = TARGET_NUM_NEGATIVES_PER_IMG):
    fileName = NEGATIVE_DATABASE_PATHS[stageIdx]
    resizeTo = SCALES[stageIdx]
    negativeImagePaths = [os.path.join(negImgFolder, fileName) for fileName in os.listdir(negImgFolder)]
    images = np.zeros((numNegatives, resizeTo[1], resizeTo[0], 3), dtype = np.uint8)
    negIdx = 0
    numNegativesRetrievedFromImg = 0

    for i in np.random.permutation(len(negativeImagePaths)):
        if negIdx >= numNegatives: break
        img = cv2.resize(cv2.imread(negativeImagePaths[i]), None, fx = resizeTo[0]/MIN_FACE_SCALE, fy = resizeTo[1]/MIN_FACE_SCALE)

        for xOffset in np.random.permutation(numDetectionWindowsAlongAxis(img.shape[1], stageIdx)):
            if negIdx >= numNegatives or numNegativesRetrievedFromImg >= numNegativesPerImg: break

            for yOffset in np.random.permutation(numDetectionWindowsAlongAxis(img.shape[0], stageIdx)):
                if negIdx >= numNegatives or numNegativesRetrievedFromImg >= numNegativesPerImg: break
                x, y, w, h = squashCoords(img, resizeTo[0]*xOffset, resizeTo[1]*yOffset, *resizeTo)

                if (w == resizeTo[0] and h == resizeTo[1]):
                    images[negIdx] = img[y:y+h,x:x+w]
                    negIdx += 1
                    numNegativesRetrievedFromImg += 1

        numNegativesRetrievedFromImg = 0

    if negIdx < len(images)-1:
        images = np.delete(images, np.s_[negIdx:], 0)

    with h5py.File(fileName, 'w') as out:
        out.create_dataset(DATASET_LABEL, data = images, chunks = (CHUNK_SIZE,) + (images.shape[1:]))

def mineNegatives(stageIdx, negImgFolder = NEGATIVE_IMAGE_FOLDER, numNegatives = TARGET_NUM_NEGATIVES):
    from .detect import detectMultiscale

    fileName = NEGATIVE_DATABASE_PATHS[stageIdx]
    resizeTo = SCALES[stageIdx]
    negativeImagePaths = [os.path.join(negImgFolder, fileName) for fileName in os.listdir(negImgFolder)]
    images = np.zeros((numNegatives, resizeTo[1], resizeTo[0], 3), dtype = np.uint8)
    negIdx = 0

    for i in np.random.permutation(len(negativeImagePaths)):
        if negIdx >= numNegatives: break
        img = cv2.imread(negativeImagePaths[i])
        coords = detectMultiscale(img, stageIdx-1)

        for xMin, yMin, xMax, yMax in coords:
            if negIdx >= numNegatives: break
            xMin, yMin, w, h = squashCoords(img, xMin, yMin, xMax-xMin, yMax-yMin)
            images[negIdx] = cv2.resize(img[yMin:yMin+h, xMin:xMin+w], resizeTo)
            negIdx += 1

    if negIdx < len(images)-1:
        images = np.delete(images, np.s_[negIdx:], 0)

    with h5py.File(fileName, 'w') as out:
        out.create_dataset(DATASET_LABEL, data = images, chunks = (CHUNK_SIZE,) + (images.shape[1:]))

def createCalibrationDataset(stageIdx, getObjectAnnotations = getFaceAnnotations, posImgFolder = POSITIVE_IMAGE_FOLDER, numCalibrationSamples = TARGET_NUM_CALIBRATION_SAMPLES, calibPatterns = CALIB_PATTERNS):
    numCalibrationSamples = math.inf if numCalibrationSamples is None else numCalibrationSamples
    objectAnnotations = getObjectAnnotations(posImgFolder = posImgFolder)

    resizeTo = SCALES[stageIdx]
    datasetLen = len(objectAnnotations)*len(calibPatterns) if numCalibrationSamples == math.inf else numCalibrationSamples
    dataset = np.zeros((datasetLen, resizeTo[1], resizeTo[0], 3), np.uint8)
    labels = np.zeros((datasetLen, 1))
    sampleIdx = 0

    fileName = CALIBRATION_DATABASE_PATHS.get(resizeTo[0])

    curImg = None
    prevImgPath = None
    
    for i, (imgPath, x, y, w, h) in enumerate(objectAnnotations):
        if sampleIdx >= numCalibrationSamples: break

        if imgPath != prevImgPath:
            curImg = cv2.imread(imgPath)

        for n, (sn, xn, yn) in enumerate(CALIB_PATTERNS):
            if sampleIdx >= numCalibrationSamples: break
            box = squashCoords(curImg, x + int(xn*w), y + int(yn*h), int(w*sn), int(h*sn))

            if box[2] > 0 and box[3] > 0:
                (box_x, box_y, box_w, box_h) = box
                dataset[sampleIdx] = cv2.resize(curImg[box_y:box_y+box_h,box_x:box_x+box_w], resizeTo)
                labels[sampleIdx] = n 
                sampleIdx += 1

    if sampleIdx < datasetLen:
        labels = np.delete(labels, np.s_[sampleIdx:], 0)
        dataset = np.delete(dataset, np.s_[sampleIdx:], 0)

    with h5py.File(fileName, 'w') as out:
        out.create_dataset(LABELS_LABEL, data = labels, chunks = (CHUNK_SIZE, 1))
        out.create_dataset(DATASET_LABEL, data = dataset, chunks = (CHUNK_SIZE, resizeTo[1], resizeTo[0], 3))

def getTestImagePaths(testImgDbFolder = TEST_IMAGE_DATABASE_FOLDER, testImgsFolder = TEST_IMAGES_FOLDER):
    imgPaths = []

    for fileName in os.listdir(testImgDbFolder):
        if 'ellipse' not in fileName:
            with open(os.path.join(testImgDbFolder, fileName)) as inFile:
                imgPaths.extend(inFile.read().splitlines())

    return [os.path.join(testImgsFolder, imgPath) + '.jpg' for imgPath in imgPaths]

class DatasetManager():
    normalizers = {}

    def __init__(self, model, getObjectAnnotations = getFaceAnnotations, posImgFolder = POSITIVE_IMAGE_FOLDER, negImgFolder = NEGATIVE_IMAGE_FOLDER):
        from .model import ObjectCalibrator
        self.model = model
        self.isCalib = isinstance(self.model, ObjectCalibrator)
        self.stageIdx = model.getStageIdx()
        self.posDatasetFilePath = OBJECT_DATABASE_PATHS[self.stageIdx]
        self.negDatasetFilePath = NEGATIVE_DATABASE_PATHS[self.stageIdx]
        self.calibDatasetFilePath = CALIBRATION_DATABASE_PATHS[SCALES[self.stageIdx][0]]
        self.posImgFolder = posImgFolder
        self.negImgFolder = negImgFolder

        self.getObjectAnnotations = getObjectAnnotations

    def getParams(self):
        return {'getObjectAnnotations': self.getObjectAnnotations, 'posImgFolder': self.posImgFolder, 'negImgFolder': self.negImgFolder}

    def _createFile(self, fileName, **kwargs):
        if not os.path.isfile(fileName) and not downloadIfAvailable(fileName):
            if fileName in OBJECT_DATABASE_PATHS:
                createPositiveDataset(self.stageIdx, **kwargs)
            elif fileName in  NEGATIVE_DATABASE_PATHS:
                (createNegativeDataset if self.stageIdx == 0 else mineNegatives)(self.stageIdx, **kwargs)
            elif fileName in CALIBRATION_DATABASE_PATHS.values():
                createCalibrationDataset(self.stageIdx, **kwargs)

    def getPosDatasetFilePath(self):
        self._createFile(self.posDatasetFilePath, posImgFolder = self.posImgFolder, getObjectAnnotations = self.getObjectAnnotations)
        return self.posDatasetFilePath

    def getNegDatasetFilePath(self):
        self._createFile(self.negDatasetFilePath, negImgFolder = self.negImgFolder)
        return self.negDatasetFilePath

    def getCalibDatasetFilePath(self):
        if self.isCalib: self._createFile(self.calibDatasetFilePath, posImgFolder = self.posImgFolder, getObjectAnnotations = self.getObjectAnnotations)
        return self.calibDatasetFilePath

    def getLabels(self):
        labels = None

        if self.isCalib:
            calibDatasetFilePath = self.getCalibDatasetFilePath()

            with h5py.File(calibDatasetFilePath, 'r') as calibDatasetFile:
                labels = calibDatasetFile[LABELS_LABEL][:]

        return labels

    def getNormalizer(self):
        from .preprocess import ImageNormalizer

        if DatasetManager.normalizers.get(self.model) is None:
            normalizer = ImageNormalizer(self.getPosDatasetFilePath(), self.getNegDatasetFilePath(), self.model.getNormalizationMethod())
            normalizer.addDataAugmentationParams(self.model.getNormalizationParams())
            DatasetManager.normalizers[self.model] = normalizer

        return DatasetManager.normalizers[self.model]

    def getPaths(self):
        return [self.getPosDatasetFilePath(), self.getNegDatasetFilePath()] if not self.isCalib else [self.getCalibDatasetFilePath(), None]