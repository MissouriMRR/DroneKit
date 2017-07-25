import os
import pickle
import numpy as np
import cv2

ROOMBA_POSITIVE_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Positives'
ROOMBA_NEGATIVE_IMAGE_FOLDER = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/Negatives'
ROOMBA_ANNOTATIONS_FILE = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/annotations'
RED_FLAP_TEMPLATE_PATH = r'/home/christopher/DroneKit/Vision/Roombas/Roomba Dataset/red_flap_template.jpg'

ARROW_LENGTH = 150

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

def squashCoords(img, x, y, w, h):
    y = min(max(0, y), img.shape[0])
    x = min(max(0, x), img.shape[1])
    h = min(img.shape[0]-y, h)
    w = min(img.shape[1]-x, w)
    return (x, y, w, h)

def getRoombaOrientations(proposals, centers):
    centers = np.asarray(centers)
    orientations = []
    
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

testRoombaImages = visualizer(getRoombaImagePaths(ROOMBA_POSITIVE_IMAGE_FOLDER), drawDirectionalArrows)
