import data
import visualize
import annotation

import cv2
import numpy as np

from data import MIN_FACE_SCALE, OFFSET
#from train import train

TWELVE_NET_FILE_NAME = '12net.hdf'
WINDOW_TITLE = '12_net_test'
TEST = True

#train(TWELVE_NET_FILE_NAME, 12)

if TEST:
    import keras
    from keras.models import load_model
    from keras.optimizers import Adam
    from train import build12net

    model = build12net()
    model.load_weights(TWELVE_NET_FILE_NAME)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    class WebcamCapture( ):
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

        def show( self, mat ):
            cv2.imshow( self.name, mat )
            return cv2.waitKey( 1 ) & 0xFF != self.quitOnKey

    def numDetectionWindows(size):
        return (size-12)//OFFSET+1

    with WebcamCapture( ) as webcam, cv2Window( WINDOW_TITLE ) as window:
        for status, frame in webcam:
            if status:
                resizedFrame = cv2.resize(frame, None, fx = 12/MIN_FACE_SCALE, fy = 12/MIN_FACE_SCALE)
                numWindows = numDetectionWindows(resizedFrame.shape[1])*numDetectionWindows(resizedFrame.shape[0])
                windows = np.ones((numWindows, 12, 12, 3), frame.dtype)
                possibleRects = []
                i = 0

                for yIdx in np.arange(numDetectionWindows(resizedFrame.shape[0])):
                    for xIdx in np.arange(numDetectionWindows(resizedFrame.shape[1])):
                        offset = np.array([xIdx * OFFSET, yIdx * OFFSET])
                        windows[i] = resizedFrame[offset[1]:offset[1]+12,offset[0]:offset[0]+12]
                        topLeft = offset*(MIN_FACE_SCALE//12)
                        bottomRight = topLeft + MIN_FACE_SCALE
                        possibleRects.insert(-1, (tuple(topLeft), tuple(bottomRight)))
                        i += 1

                predictions = model.predict(windows).argmax(axis=1)

                for j in np.arange(len(possibleRects)):
                    if bool(predictions[j]):
                        cv2.rectangle(frame, possibleRects[j][0], possibleRects[j][1], (0, 255, 0), 1)
    
                if not window.show( frame ):
                    break
