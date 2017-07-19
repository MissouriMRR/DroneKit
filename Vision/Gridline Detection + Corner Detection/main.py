import cv2
import numpy as np
import os

from timeit import default_timer as timer

DEFAULT_DELAY_BETWEEN_FRAMES_IN_MS = 25
VIDEO_FILE_NAME = 'IARC 2016 Footage.mp4'
WINDOW_TITLE = 'Gridline/Corner Detection Test'
QUIT_KEY = 'q'

INTENSITY_THRESH_VAL = 180
LINE_THRESHOLD_VAL = 75
MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 1000
EPSILON = .05
MIN_RECT_AREA = 500
PERCENT_CONTOUR_ARC_LENGTH = .15

RED = (0, 0, 255)
GREEN = (0, 255, 0)

LINE_THICKNESS = 20
CIRCLE_THICKNESS = 5
CIRCLE_RADIUS = 100

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

class VideoReader( ):
  def __init__(self, fileName):
    self.fileName = fileName

  def __enter__(self):
    self.cap = cv2.VideoCapture(self.fileName)
    return self

  def next(self):
    if (not self.cap.isOpened()):
      return None

    return self.cap.read()[1]

  def __exit__(self, *args):
    self.cap.release()

def detectLinesAndCorners(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  thresholded = cv2.threshold(gray, INTENSITY_THRESH_VAL, 255, cv2.THRESH_BINARY_INV)[1]
  thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, np.ones((5, 5),np.uint8))
  edges = cv2.Canny(thresholded,150,200, apertureSize = 5)
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, LINE_THRESHOLD_VAL, minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP)
  lineCoords = []

  if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        lineCoords.append((x1, y1, x2, y2))
        cv2.line(edges, (x1,y1), (x2, y2), 255, LINE_THICKNESS)

  image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  rects = []
  rectCentroids = []

  if contours is not None:
    for i, contour in enumerate(contours):
      approx = cv2.approxPolyDP(contour, PERCENT_CONTOUR_ARC_LENGTH*cv2.arcLength(contour, True), True)
      x, y, w, h = cv2.boundingRect(approx)
      aspectRatio = w / float(h)
      numVertices = len(approx)

      if hierarchy[0][i][2] < 0 and numVertices == 4 and np.abs(aspectRatio-1) <= EPSILON and w*h >= MIN_RECT_AREA:
        rects.append(contour)

    if len(rects) > 0:
      for rect in rects:
        moments = cv2.moments(rect)
        x, y = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        rectCentroids.append((x, y))

    return lineCoords, rectCentroids

with VideoReader(VIDEO_FILE_NAME) as video, cv2Window(WINDOW_TITLE) as window:
  frame = video.next()
  quit = False

  while frame is not None and not quit:
    lineCoords, rectCentroids = detectLinesAndCorners(frame)

    for (x1, y1, x2, y2) in lineCoords:
      cv2.line(frame, (x1, y1), (x2, y2), RED, LINE_THICKNESS)
    for (x, y) in rectCentroids:
      cv2.circle(frame, (x, y), CIRCLE_RADIUS, GREEN, CIRCLE_THICKNESS)

    window.show(frame)

    start = timer()
    while (timer()-start) <= DEFAULT_DELAY_BETWEEN_FRAMES_IN_MS/1000:
      if window.isKeyDown( QUIT_KEY ):
        quit = True
        break

    frame = video.next()