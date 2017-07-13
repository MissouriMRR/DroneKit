from ImageDatasetUtility import WORKING_DIR, workingdir
from time import sleep
import cv2
import numpy as np
import os
import pickle

SAVE_FILE_NAME = 'annotations'

WINDOW_NAME = 'Annotator' 
RECT_COLOR = (0, 255, 0)
RECT_AREA_THRESHOLD = 5
THICKNESS = 2
DELAY = 1

class cv2Window( ):
    def __init__( self, name, type = cv2.WINDOW_AUTOSIZE ):
        self.name = name
        self.type = type

    def __enter__( self ):
        cv2.namedWindow( self.name, self.type )
        return self

    def __exit__( self, *args ):
        cv2.destroyWindow( self.name )

    def setMouseCallback( self, func, param = None ):
        cv2.setMouseCallback( WINDOW_NAME, func, param )

    def show( self, mat ):
        cv2.imshow( self.name, mat )

    def getKeyDown( self ):
        return chr( cv2.waitKey( DELAY ) & 0xFF )

    def isKeyPressed( self, key ):
        return ( cv2.waitKey( DELAY ) & 0xFF ) == ord( key )


def mouseEvent( event, x, y, flags, annotator ):
    if event == cv2.EVENT_LBUTTONDOWN:
        annotator.lbuttondown( x, y )
    elif event == cv2.EVENT_MOUSEMOVE:
        annotator.mouseMove( x, y )
    elif event == cv2.EVENT_LBUTTONUP:
        annotator.lbuttonup( x, y )

class Annotations( ):
    def __init__( self ):
        savedfileName = os.path.join( WORKING_DIR, SAVE_FILE_NAME )

        if not os.path.isfile( savedfileName ):
            self.annotations = {}
        else:
            with open( savedfileName, 'rb' ) as inFile:
                self.annotations = pickle.load( inFile )

    def store( self, fileName, annotations ):
        self.annotations[fileName] = annotations

    def clear( self, fileName ):
        if self.has( fileName ):
            del self.annotations[fileName]

    def has( self, fileName ):
        return fileName in self.annotations

    def get( self, fileName ):
        return self.annotations[fileName] if self.has( fileName ) else []

    def save( self, asFileName ):
        with open( asFileName, 'wb' ) as outFile:
            pickle.dump( self.annotations, outFile )

@workingdir
def _mainloop( annotator ):
    events = {
        'n' : lambda: annotator.next(),
        'p' : lambda: annotator.prev(),
        'c' : lambda: annotator.clear(),
    }

    while True:
        key = annotator.dialog.getKeyDown( )

        if key == 'q':
            break
        elif key in events:
            events[key]()

    annotator.Annotations.save( SAVE_FILE_NAME )

IMAGE_FILE_EXTENSIONS = {}.fromkeys( ( '.jpg', '.png' ), True )

class Annotator():
    def __init__( self ):
        self.Annotations = Annotations( )
        self.idx = 0
        self.imgFileNames = [name for name in os.listdir( WORKING_DIR ) if os.path.splitext(name)[1] in IMAGE_FILE_EXTENSIONS]
        self.numImages = len( self.imgFileNames )

        self.curFileName = self.imgFileNames[self.idx]
        self.curImg = cv2.imread( os.path.join( WORKING_DIR, self.curFileName ) ) if self.imgFileNames else None
        self.curAnnotations = self.Annotations.get( self.curFileName )

        self.rect = np.zeros((2, 2), dtype=np.int)
        self.lbuttonDown = False

    def loop( self ):
        assert self.curImg is not None, 'No images available to annotate in the current working directory!'

        with cv2Window( WINDOW_NAME ) as dialog:
            self.dialog = dialog
            self.dialog.setMouseCallback( mouseEvent, self )
            self._update( )
            _mainloop( self )

    def _drawRect( self, img, mat ):
        cv2.rectangle( img, tuple(mat[:,0]), tuple(mat[:,1]), RECT_COLOR, THICKNESS )

    def _setRectCoord( self, x, y, idx ):
        self.rect[:,idx] = np.array((x,y), dtype=np.int)

    def _update( self, rect = None ):
        tmp = self.curImg.copy()

        if ( rect is not None ):
            self._drawRect( tmp, rect )

        for annotation in self.curAnnotations:
            self._drawRect( tmp, annotation )

        self.dialog.show( tmp )
     
    def _setIdx( self, val ):
        if val >= self.numImages:
            self.idx = val % self.numImages
        elif val < 0:
            self.idx = self.numImages - 1
        else:
            self.idx = val
        
        self.Annotations.store( self.curFileName, self.curAnnotations )

        self.curFileName = self.imgFileNames[self.idx]
        self.curImg = cv2.imread( self.curFileName )
        self.curAnnotations = self.Annotations.get( self.curFileName )

    def next( self ):
        self._setIdx( self.idx + 1 )
        self._update( )

    def prev( self ):
        self._setIdx( self.idx - 1 )
        self._update( )

    def clear( self ):
        self.curAnnotations = []
        self.Annotations.clear( self.curFileName )
        self._update( )
    
    def lbuttondown( self, x, y ):
        self.lbuttonDown = True
        self._setRectCoord( x, y, 0 )

    def mouseMove( self, x, y ):
        if ( self.lbuttonDown ):
            self._setRectCoord( x, y, 1 )
            self._update( self.rect )

    def lbuttonup( self, x, y ):
        self.lbuttonDown = False
        self._setRectCoord( x, y, 1 )

        if np.abs(np.linalg.det( self.rect )) > RECT_AREA_THRESHOLD:
            self.curAnnotations.insert( -1, self.rect.copy() )
            self.Annotations.store(self.curFileName, self.curAnnotations)
            self._update()