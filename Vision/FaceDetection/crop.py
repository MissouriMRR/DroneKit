import cv2
import numpy as np

VALID_CROP_THRESHOLD = .75

def crop(mat, topLeft, bottomRight, newW = None, newH = None):
    fitCoordToImage = lambda coord: (min(max(0, coord[0]),mat.shape[1]-1), min(max(0, coord[1]), mat.shape[0]-1))
    top_left = fitCoordToImage(topLeft)
    bottom_right = fitCoordToImage(bottomRight)
    cropped = None
        
    ratio = lambda dim: (bottom_right[dim]-top_left[dim])/(bottomRight[dim]-topLeft[dim]) if (bottomRight[dim]-topLeft[dim]) > 0 else 0

    if (ratio(0) >= VALID_CROP_THRESHOLD and ratio(1) >= VALID_CROP_THRESHOLD):
        region = mat[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        croppedSize = (newW if newW else region.shape[1], newH if newH else region.shape[0])
        cropped = cv2.resize(region, croppedSize)

    return cropped

def cropOutROIs(images, w, h):
    db = []

    for i, imgPath in enumerate(images.keys()):
        img = cv2.imread(imgPath)

        for annotation in images.getAnnotations(imgPath):
            cropped = annotation.cropOut(img, w, h)
            if cropped is None: continue
            db.insert(-1, cropped)

    arr = np.ones((len(db), w, h, 3), dtype=img.dtype)

    for i, img in enumerate(db):
        arr[i] = img

    return arr