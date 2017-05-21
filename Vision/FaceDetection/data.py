import h5py
import random
import cv2
import numpy as np
import os

from crop import crop, cropOutROIs
from annotation import RectangleAnnotation, EllipseAnnotation, getFaceAnnotations, ANNOTATIONS_FOLDER, POSITIVE_IMAGES_FOLDER

NEGATIVE_IMAGES_FOLDER = r'./Negative Images/images'
FACE_DATABASE_PATHS = ('face12.hdf', 'face24.hdf', 'face48.hdf')
NEGATIVE_DATABASE_PATHS = ('neg12.hdf', 'neg24.hdf', 'neg48.hdf')
TRAIN_DATABASE_PATH = 'train.hdf'
NUM_NEGATIVES_PER_IMG = 20
MIN_FACE_SCALE = 50
OFFSET = 4

SCALES = ((12,12),(24,24),(48,48))

CALIBRATION_DATABASE_PATHS = {SCALES[0][0]:'calib12.hdf',SCALES[1][0]:'calib24.hdf',SCALES[2][0]:'calib48.hdf'}
TARGET_NUM_CALIBRATION_SAMPLES = 5000
SN = (.83, .91, 1, 1.1, 1.21)
XN = (-.17, 0, .17)
YN = XN

def loadDatabase(databasePath, isNegativeDataset = False):
    db = None

    try:
        with h5py.File(databasePath, 'r') as infile:
            db = infile[databasePath[:databasePath.find('.')]][:]
    except IOError as ioError:
        pass

    return db

def getNegatives(w, h, imgFolder = NEGATIVE_IMAGES_FOLDER, numNegativesPerImg =  NUM_NEGATIVES_PER_IMG, numNegatives = None):
    posDb = loadDatabase('face%d.hdf' % w)
    numNegatives = posDb.shape[0]*NUM_NEGATIVES_PER_IMG if posDb is not None and numNegatives is None else numNegatives
    db = np.ones((numNegatives,w,h,3),dtype=posDb[0].dtype)
    len = 0

    for i, imgPath in enumerate(os.listdir(imgFolder)):
        if len >= int(numNegatives*.9): break
        img = cv2.imread('%s\%s' % ( NEGATIVE_IMAGES_FOLDER, imgPath ))
        maxDimIdx = ((img.shape[1]-MIN_FACE_SCALE)//MIN_FACE_SCALE,(img.shape[0]-MIN_FACE_SCALE)//MIN_FACE_SCALE)
        randXIdx = np.random.permutation(maxDimIdx[0])
        randYIdx = np.random.permutation(maxDimIdx[1])
        checked = 0
        extracted = 0

        for j in np.arange(min(randXIdx.shape[0], randYIdx.shape[0])):
            if len >= int(numNegatives*.9) or checked >= randXIdx.shape[0]*randYIdx.shape[0] or extracted >= NUM_NEGATIVES_PER_IMG: break
            top_left = (randXIdx[j] * MIN_FACE_SCALE, randYIdx[j] * MIN_FACE_SCALE)
            bottom_right = (top_left[0] + MIN_FACE_SCALE, top_left[1] + MIN_FACE_SCALE)
            cropped = crop(img, top_left, bottom_right, MIN_FACE_SCALE, MIN_FACE_SCALE)
            if cropped is None: 
                continue
            db[len] = cv2.resize(cropped, (w,h))
            len += 1
            extracted += 1

    faceAnnotations = getFaceAnnotations()

    for i, imgPath in enumerate(faceAnnotations.keys()):
       if len >= numNegatives: break
       annotations = faceAnnotations.getAnnotations(imgPath)
       img = cv2.imread(imgPath)
       maxDimIdx = ((img.shape[1]-MIN_FACE_SCALE)//MIN_FACE_SCALE,(img.shape[0]-MIN_FACE_SCALE)//MIN_FACE_SCALE)
       randXIdx = np.random.permutation(maxDimIdx[0])
       randYIdx = np.random.permutation(maxDimIdx[1])
       checked = 0
       extracted = 0

       for j in np.arange(min(randXIdx.shape[0], randYIdx.shape[0])):
            if len >= numNegatives or checked >= randXIdx.shape[0]*randYIdx.shape[0] or extracted >= NUM_NEGATIVES_PER_IMG: break
            checked += 1
            top_left = (randXIdx[j] * MIN_FACE_SCALE, randYIdx[j] * MIN_FACE_SCALE)
            bottom_right = (top_left[0] + MIN_FACE_SCALE, top_left[1] + MIN_FACE_SCALE)
            rect = RectangleAnnotation(MIN_FACE_SCALE, MIN_FACE_SCALE, top_left[0] + MIN_FACE_SCALE//2, top_left[1] + MIN_FACE_SCALE//2)
            if any([annotation.computeIoU(rect) > 0 for annotation in annotations]):
               continue
            cropped = crop(img, top_left, bottom_right, MIN_FACE_SCALE, MIN_FACE_SCALE)
            if cropped is None: continue
            db[len] = cv2.resize(cropped, (w,h))
            len += 1
            extracted += 1

    return db

def createDatabase(databasePaths, loadFunc, scales = SCALES):
    for i, databasePath in enumerate(databasePaths):
        with h5py.File(databasePath, 'w') as out:
            w, h = SCALES[i]
            out.create_dataset(databasePath[:databasePath.find('.')], data = loadFunc(w,h), chunks=(32,w,h,3))

def createNegativeDatabase():
    createDatabase(NEGATIVE_DATABASE_PATHS, lambda w, h: getNegatives(w,h))

def createFaceDatabase(faces):
    createDatabase(FACE_DATABASE_PATHS, lambda w, h, faces = faces: cropOutROIs(faces, w, h))

def _createCalibrationDataset(faces, scale = SCALES[0][0], numCalibrationSamples = TARGET_NUM_CALIBRATION_SAMPLES):
    imgDtype = loadDatabase('face%d.hdf' % scale)[0].dtype
    numCalibPatterns = len(SN)*len(XN)*len(YN)
    calibDbLen = numCalibrationSamples * numCalibPatterns

    db = np.ones((calibDbLen, scale, scale, 3), imgDtype)
    y = np.ones((calibDbLen,1))

    i = 0
    j = 0
    posImgPaths = tuple(faces.keys())
    calibPattterns = [(sn, xn, yn) for sn in SN for xn in XN for yn in YN]

    with h5py.File(CALIBRATION_DATABASE_PATHS.get(scale), 'w') as out:
        while i < calibDbLen and j < len(posImgPaths):
            img = cv2.imread(posImgPaths[j])
            
            for annotation in faces.getAnnotations(posImgPaths[j]):
                for n, (sn, xn, yn) in enumerate(calibPattterns):
                    dim = np.array([annotation.w, annotation.h])
                    top_left = annotation.top_left + (np.array([xn, yn])*dim).astype(int)
                    dim = (dim*sn).astype(int)
                    cropped = crop(img, top_left, top_left + dim, scale, scale)

                    if cropped is not None:
                        db[i] = cropped
                        y[i] = n
                        i += 1
            j += 1
        
        if i < calibDbLen:
            np.delete(y, np.s_[i:], 0)
            np.delete(db, np.s_[i:], 0)

        out.create_dataset('labels', data=y, chunks=(32,1))
        out.create_dataset('data', data=db, chunks=(32, scale, scale, 3))

def createCalibrationDatabase(scale = SCALES):
    faces = getFaceAnnotations()
    for (w, h) in SCALES:
        _createCalibrationDataset(faces, w)


def createTrainingDataset(scale = SCALES[0][0]):
    pos_db = loadDatabase('face%d.hdf' % scale)
    neg_db = loadDatabase('neg%d.hdf' % scale)
    img_dtype = pos_db[0].dtype
    y = np.vstack((np.ones((len(pos_db),1),dtype=img_dtype),np.zeros((len(neg_db),1))))
    db = np.vstack((pos_db,neg_db))
    
    perm = np.random.permutation(db.shape[0])
    y = y[perm]
    db = db[perm]

    with h5py.File(TRAIN_DATABASE_PATH, 'w') as out:
        out.create_dataset('labels', data=y, chunks=(32,1))
        out.create_dataset('data', data=db, chunks=(32, scale, scale, 3))
