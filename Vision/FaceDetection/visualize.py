#!/usr/bin/env python3.5
import numpy as np
import cv2

from data import SCALES

class cv2Window( ):
    def __init__( self, name, type = cv2.WINDOW_AUTOSIZE ):
        self.name = name
        self.type = type

    def __enter__( self ):
        cv2.namedWindow( self.name, self.type )
        return self

    def __exit__( self, *args ):
        cv2.destroyWindow( self.name )

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
