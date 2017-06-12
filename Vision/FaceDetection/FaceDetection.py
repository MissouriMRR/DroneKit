#!/usr/bin/env python3.5
from timeit import default_timer as timer

WINDOW_TITLE = 'Face Detector Test'
TEST = False

TRAIN = True
TRAIN_CALIB = True

PROFILE = True
DEBUG = False

STAGE_IDX = 1

GREEN = (0, 255, 0)
THICKNESS = 3

if __name__ == '__main__':
    import cv2
    import data
    from visualize import visualizer
    from detect import detectMultiscale

    data.createCalibrationDataset(STAGE_IDX)
    
    def predictionCallback(img):
        start = timer()
        detections = detectMultiscale(img)
        if PROFILE:
            print('Prediction took %fs' % (timer() - start,))
        
        for (xMin, yMin, xMax, yMax) in detections: 
            cv2.rectangle(img, (xMin, yMin), (xMax, yMax), GREEN, THICKNESS)


    if TEST:
        visualizer(data.getTestImagePaths(), predictionCallback, WINDOW_TITLE)
    elif TRAIN:
        from train import train
        train(STAGE_IDX, TRAIN_CALIB)
        