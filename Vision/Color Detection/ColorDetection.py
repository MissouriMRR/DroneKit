import numpy as np
import contextlib
import cv2

WINDOW_TITLE = 'Press space to take a picture!' 

trackbars = {}.fromkeys( ( 'min h', 'max h', 'min s', 'max s', 'min v', 'max v' ), 0 )

def nothing( x ): print( x )

class WebcamCapture( contextlib.AbstractContextManager ):
    def __init__( self, devId = 0 ):
        self.devId = devId

    def __enter__( self ):
        self.cap = cv2.VideoCapture( self.devId )
        return self

    def __exit__( self, *args ):
        self.cap.release( )

    def __iter__( self ):
        return self

    def __next__( self ):
        if not self.cap.isOpened( ):
            raise StopIteration
        
        return self.cap.read( )

class cv2Window( ):
    def __init__( self, name, type = cv2.WINDOW_AUTOSIZE, quitOnKey = 'q' ):
        self.name = name
        self.type = type
        self.quitOnKey = ord( quitOnKey )

    def __enter__( self ):
        cv2.namedWindow( self.name, self.type )
        return self

    def __exit__( self, *args ):
        cv2.destroyWindow( self.name )

    def addTrackbar( self, name, min, max, callback = nothing ):
        cv2.createTrackbar( name, self.name, min, max, callback )

    def show( self, mat ):
        cv2.imshow( self.name, mat )
        return cv2.waitKey( 1 ) & 0xFF != self.quitOnKey

def update( trackbarName, val ):
    trackbars[trackbarName] = val

def thresholdFrameByHSV( frame ):
    min = np.array( [trackbars['min h'], trackbars['min s'], trackbars['min v']], np.uint8 )
    max = np.array( [trackbars['max h'], trackbars['max s'], trackbars['max v']], np.uint8 )

    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
    mask = cv2.inRange( hsv, min, max )

    return cv2.bitwise_and( frame, frame, mask = mask )

MIN_RED = np.array( [0, 203, 0], np.uint8 )
MAX_RED = np.array( [8, 255, 157], np.uint8 )

def detectRedObject( frame ):
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
    mask = cv2.inRange( hsv, MIN_RED, MAX_RED )

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    if contours:
        largestContour = max( contours, key = cv2.contourArea )

        rect = cv2.minAreaRect( largestContour )
        box = cv2.boxPoints( rect )
        box = np.int0( box )
        cv2.drawContours( frame, [box], 0, (0,255, 0), 3 )


with WebcamCapture( ) as webcam, cv2Window( WINDOW_TITLE ) as window:
    #for trackbarName in trackbars:
    #    window.addTrackbar( trackbarName, 0, 180 if 'h' in trackbarName else 255, lambda val, trackbarName = trackbarName: update( trackbarName, val ) )

    for status, frame in webcam:
        if status:
            detectRedObject( frame )

            if not window.show( frame ):
                break

