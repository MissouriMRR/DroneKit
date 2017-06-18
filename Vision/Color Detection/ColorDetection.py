import numpy as np
import cv2

from RealSense import Streamer, LiveDisplay

WINDOW_TITLE = 'Color Detection Test' 
EXIT_KEY = 'q'

TEST = True

TRACKBAR_NAMES = ( 'min h', 'max h', 'min s', 'max s', 'min v', 'max v' )
trackbars = {}.fromkeys( TRACKBAR_NAMES, 0 )

def update( val, trackbarName ):
    trackbars[trackbarName] = val

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

    def addTrackbar( self, name, min, max, callback = lambda *args: None ):
        cv2.createTrackbar( name, self.name, min, max, callback )

    def isKeyDown(self, key):
        return cv2.waitKey( 1 ) & 0xFF == ord(key)

    def getKey(self):
        return chr(cv2.waitKey( 1 ) & 0xFF)

    def show( self, mat ):
        cv2.imshow( self.name, mat )

def thresholdFrameByHSV( frame ):
    min = np.array( [trackbars['min h'], trackbars['min s'], trackbars['min v']], np.uint8 )
    max = np.array( [trackbars['max h'], trackbars['max s'], trackbars['max v']], np.uint8 )

    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
    mask = cv2.inRange( hsv, min, max )

    return cv2.bitwise_and( frame, frame, mask = mask )

MIN_RED = np.array( [0, 165, 54], np.uint8 )
MAX_RED = np.array( [180, 255, 171], np.uint8 )

def detectRedObject( frame ):
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
    mask = cv2.inRange( hsv, MIN_RED, MAX_RED )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8) )

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    if contours:
        largestContour = max( contours, key = cv2.contourArea )

        rect = cv2.minAreaRect( largestContour )
        box = cv2.boxPoints( rect )
        box = np.int0( box )
        cv2.drawContours( frame, [box], 0, (0,255, 0), 3 )


with cv2Window(WINDOW_TITLE) as win, Streamer() as stream:
    liveStream = LiveDisplay(stream, win)
    if TEST:
        liveStream.run(lambda frame: detectRedObject(frame))
    else:
        for name in TRACKBAR_NAMES: 
            win.addTrackbar(name, 0, 255 if 'h' not in name else 180, lambda val, name = name: update(val, name))

        liveStream.run(lambda frame: thresholdFrameByHSV(frame))