from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import base64
import cv2
import numpy as np
import pickle

from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect

from .data import squashCoords
from .detect import fastDetect
from .visualize import cv2Window
from .task import MIN_AREA, THRESHOLD_MAX, THRESHOLD_MIN, ARROW_LENGTH, GREEN, THICKNESS

PORT_NUMBER = 8888
LOG_PREFIX = 'CLIENT:'
WINDOW_TITLE = 'Client'
KEEP_ALIVE_DELAY = 20000

def getRoombaOrientations(img, proposals, centers):
    centers = np.asarray(centers)
    orientations = []

    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, binaryFlaps = cv2.threshold(hsvImage[:,:,1], THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(img, np.repeat(binaryFlaps[:, :, np.newaxis], 3, 2))

    for i, (xMin, yMin, xMax, yMax) in enumerate(proposals):
        x, y, w, h = squashCoords(img, xMin, yMin, xMax - xMin, yMax - yMin)
        closing = cv2.morphologyEx(img[y:y+h, x:x+w], cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=3)
        gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, np.ones((7,7),np.uint8))
        gray = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        yVec, xVec = np.nonzero(thresh)
        
        if len(xVec) > 0 and len(yVec) > 0:
            xVec = xVec - np.mean(xVec)
            yVec = yVec - np.mean(yVec)
            coords = np.vstack([xVec, yVec])
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov)
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
            maxSum = 0
            foundSign = None
            foundEvec = None

            for evec in evecs:
                for sign in (1, -1):
                    x0, y0 = centers[i]
                    x1, y1 = centers[i]+sign*evec*ARROW_LENGTH
                    x1 = min(max(x, x1), (x+w-1))
                    y1 = min(max(y, y1), (y+h-1))
                    length = int(np.hypot(x1-x0, y1-y0))
                    xIdxs, yIdxs = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
                    intensitySum = np.sum(grayImage[yIdxs.astype(int), xIdxs.astype(int)])

                    if intensitySum >= maxSum:
                        maxSum = intensitySum
                        foundSign = sign
                        foundEvec = evec
            
            orientations.append(foundEvec * foundSign)

    return orientations

def getRoombaProposals(img):
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

class OffboardClient():
    def __init__(self):
        self.url = 'ws://127.0.0.1:%d/ws' % (PORT_NUMBER,)
        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()

    def start(self):
        self.ioloop.start()

    def __enter__(self):
        self.window = cv2Window(WINDOW_TITLE).__enter__()
        return self

    def __exit__(self, *args):
        self.window.__exit__()

    def log(self, *args):
        print(LOG_PREFIX, *args)

    @gen.coroutine
    def connect(self):
        try:
            self.ws = yield websocket_connect(self.url)
        except Exception as e:
            self.log(str(e))
        else:
            self.log('Connected to server.')
            self.run()

    @gen.coroutine
    def run(self):
        while True:
            msg = yield self.ws.read_message()
     
            if msg is None:
                self.log('Shutting down client...')
                
            img = pickle.loads(msg)
            detections, centers = fastDetect(img, getRoombaProposals)
            orientations = getRoombaOrientations(img, detections, centers)

            for i, (xMin, yMin, xMax, yMax) in enumerate(detections): 
                cv2.rectangle(img, (xMin, yMin), (xMax, yMax), GREEN, THICKNESS)
                cv2.circle(img, tuple(centers[i]), 5, (255, 0, 0), 3)
                cv2.arrowedLine(img, tuple(centers[i]), tuple((centers[i]+orientations[i]*ARROW_LENGTH).astype(int)), (255, 0, 0), 3)

            self.window.show(img)

            if self.window.getKey() == 'q':
                self.ws.close()
                cv2.destroyAllWindows()
                exit()
            
            self.ws.write_message(pickle.dumps({'orientations': orientations, 'centers': centers}, 2), binary = True)


def start_client():
    with OffboardClient() as client:
        client.start()