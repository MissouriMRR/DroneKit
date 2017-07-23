from __future__ import absolute_import, division, print_function, unicode_literals
from optparse import OptionParser
from timeit import default_timer as timer

WINDOW_TITLE = 'Roomba Detector Test'
LIVE_WINDOW_TITLE = 'RealSense Test'

PROFILE = True
DEBUG = False

GREEN = (0, 255, 0)
THICKNESS = 3

ROOMBA_POSITIVE_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Positives'
ROOMBA_NEGATIVE_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Negatives'
ROOMBA_ANNOTATIONS_FILE = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/annotations'

if __name__ == '__main__':
    from .data import DatasetManager
    from .model import MODELS

    import os
    import numpy as np
    import pickle
    import cv2

    parser = OptionParser()
    parser.add_option('-s', '--stage', dest='stageIdx', help='Cascade stage index', metavar = '[0-2]', default = 2)
    parser.add_option('-c', '--calib', action='store_true', dest = 'trainCalib', help='Use to train the calibration net for the given stage', default = False)
    parser.add_option('-v', '--visualize',  action='store_true', dest = 'testMode', help='View output of detector on different test images', default = False)
    parser.add_option('-l', '--live',  action='store_true', dest = 'liveMode', help='Test detector on a live Intel RealSense Stream', default = False)
    parser.add_option('-e', '--eval', action='store_true', dest='evalMode', help="Plot preicison+recall vs. threshold for given stage", default = False)
    parser.add_option('-t', '--train', action='store_true', dest='trainMode', help='Train either the classifier or calibrator for the given stage', default = False)

    (options, args) = parser.parse_args()

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
                        roombas.append((curFilePath, top_x, top_y, np.abs(top_x - bottom_x), np.abs(top_y - bottom_y)))

        return roombas

    def getRoombaProposals(img):
        THRESHOLD_MIN = 130
        THRESHOLD_MAX = 255
        MIN_AREA = 50

        proposals = []
        centers = []

        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ret, thresholded = cv2.threshold(hsvImage[:,:,1], THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)
        erosion = cv2.erode(thresholded, np.ones((11,11), np.uint8))
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8), iterations = 5)
        
        modifiedImg, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= MIN_AREA:
                moments = cv2.moments(contour)
                centers.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
                x, y, w, h = cv2.boundingRect(contour)
                dimensions = np.array([w, h])
                topLeft = np.array([x, y]) - MIN_AREA
                bottomRight = np.array([x, y]) + dimensions + MIN_AREA
                proposals.append((*tuple(topLeft.astype(int)), *tuple(bottomRight.astype(int))))
            
        return proposals, centers
    
    def getRoombaImagePaths(posImgFolder = ROOMBA_POSITIVE_IMAGE_FOLDER):
        return [os.path.join(posImgFolder, fileName) for fileName in os.listdir(posImgFolder)]

    if options.trainMode or options.evalMode:
        model = MODELS[options.trainCalib][int(options.stageIdx)]
        datasetManager = DatasetManager(model, getObjectAnnotations = getRoombaAnnotations, posImgFolder = ROOMBA_POSITIVE_IMAGE_FOLDER, negImgFolder = ROOMBA_NEGATIVE_IMAGE_FOLDER)

    def predictionCallback(img):
        import cv2
        from .detect import fastDetect
        start = timer()
        detections, centers = fastDetect(img, getRoombaProposals)

        if PROFILE:
            print('Prediction took %fs' % (timer() - start,))
        
        for i, (xMin, yMin, xMax, yMax) in enumerate(detections): 
            cv2.rectangle(img, (xMin, yMin), (xMax, yMax), GREEN, THICKNESS)
            cv2.circle(img, tuple(centers[i]), 5, (255, 0, 0), 3)

    if options.testMode:
        from .visualize import visualizer
        from .data import getTestImagePaths
        visualizer(getRoombaImagePaths(), predictionCallback, WINDOW_TITLE)
    elif options.trainMode:
        from .train import train
        train(model, datasetManager)
    elif options.liveMode:
        from visualize import cv2Window
        from RealSense import Streamer, LiveDisplay

        with cv2Window(LIVE_WINDOW_TITLE) as win, Streamer() as stream:
            liveStream = LiveDisplay(stream, win)
            liveStream.run(predictionCallback)
    elif options.evalMode:
        from .eval import ModelEvaluator
        
        with ModelEvaluator(model, datasetManager) as evaluator:
            evaluator.summary()
