import os
import pickle
import numpy as np
import cv2

ROOMBA_POSITIVE_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Positives'
ROOMBA_NEGATIVE_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Negatives'
ROOMBA_ORIENTATION_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Orientation'
ROOMBA_ANNOTATIONS_FILE = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/annotations'

def squashCoords(img, x, y, w, h):
    y = min(max(0, y), img.shape[0])
    x = min(max(0, x), img.shape[1])
    h = min(img.shape[0]-y, h)
    w = min(img.shape[1]-x, w)
    return (x, y, w, h)

def getRoombaAnnotations(annotationsFilePath = ROOMBA_ANNOTATIONS_FILE, posImgFolder = ROOMBA_POSITIVE_IMAGE_FOLDER):
    roombas = []

    with open(annotationsFilePath, 'rb') as annotationsFile:
        annotations = pickle.load(annotationsFile)

        for curFileName, curFileAnnotations in annotations.items():
            curFilePath = os.path.join(posImgFolder, curFileName)

            if os.path.isfile(curFilePath):
                for annotation in curFileAnnotations:
                    top_x, top_y = tuple(annotation[:,0])
                    bottom_x, bottom_y = tuple(annotation[:, 1])
                    roombas.append((curFilePath, top_x, top_y, bottom_x - top_x, bottom_y - top_y))

    return roombas

def getRoombaImagePaths(posImgFolder = ROOMBA_POSITIVE_IMAGE_FOLDER):
    return [os.path.join(posImgFolder, fileName) for fileName in os.listdir(posImgFolder)]

def writeRoombaImagesToFolder(folderPath = ROOMBA_ORIENTATION_IMAGE_FOLDER):
    objectAnnotations = getRoombaAnnotations()

    for (imgPath, x, y, w, h) in objectAnnotations:
        curImg = None
        prevImgPath = None

        if imgPath != prevImgPath:
            curImg = cv2.imread(imgPath)

        x, y, w, h = squashCoords(curImg, x, y, w, h)
        cv2.imwrite(os.path.join(folderPath, os.path.basename(imgPath)), curImg[y:y+h, x:x+w])
        prevImgPath = imgPath

writeRoombaImagesToFolder()


