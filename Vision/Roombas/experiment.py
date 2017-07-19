import numpy as np
import cv2
import os
import math

ROOMBA_POSITIVES_FOLDER_PATH = 'Roomba Dataset/Positives'

class cv2Window( ):
    def __init__( self, name, type = cv2.WINDOW_AUTOSIZE ):
        self.name = name
        self.title = name
        self.type = type

    def __enter__( self ):
        cv2.namedWindow( self.name, self.type )
        return self

    def __exit__( self, *args ):
        cv2.destroyWindow( self.name )

    def getTitle(self):
        return self.title

    def setTitle(self, new_title):
        self.title = new_title
        cv2.setWindowTitle(self.name, self.title)

    def isKeyDown(self, key):
        return cv2.waitKey( 1 ) & 0xFF == ord(key)

    def getKey(self):
        return chr(cv2.waitKey( 1 ) & 0xFF)

    def show( self, mat ):
        cv2.imshow( self.name, mat )

def visualizer(images, callback = None, win_title = 'Visualizer'):
    quit = False
    length = len(images)
    i = 0
    img = None

    with cv2Window( win_title ) as window:
        while not quit:
            if type(images[i]) is np.ndarray:
                img = images[i]
            elif type(images[i]) is str:
                img = cv2.imread(images[i])

            if callback:
                callback(img)

            window.show(img)
            key = window.getKey()

            while key not in 'npq':
                key = window.getKey()

            if key == 'n':
                i = ( i + 1 ) % length
            elif key == 'p':
                i = i - 1 if i > 0 else length-1
            elif key == 'q':
                quit = True

def getRoombaProposals(img):
    THRESHOLD_MIN = 125
    THRESHOLD_MAX = 255
    MIN_AREA = 50

    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, thresholded = cv2.threshold(hsvImage[:,:,1], THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)
    erosion = cv2.erode(thresholded, np.ones((11,11), np.uint8))
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8), iterations = 5)
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = MIN_AREA
    params.maxArea = math.inf
    params.filterByColor = False
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(closing)
    print(len(keypoints))

    imgWithProposalsShown = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('test', imgWithProposalsShown)

roombaImagePaths = [os.path.join(ROOMBA_POSITIVES_FOLDER_PATH, fileName) for fileName in os.listdir(ROOMBA_POSITIVES_FOLDER_PATH)]
visualizer(roombaImagePaths, getRoombaProposals)