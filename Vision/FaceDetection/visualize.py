import cv2
import numpy as np
import h5py
import data

from data import TRAIN_DATABASE_PATH, SCALES

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

def visualizer(img_db, callback = None, win_title = 'Visualizer'):
    quit = False

    with cv2Window( win_title ) as window:
        for i in np.arange(img_db.shape[0]):
            if quit: break
            img = img_db[i]

            if callback: 
                callback(img, img_db, imgPath)

            window.show(img)
            key = window.getKey()

            while not ( quit or key == 'n' ):
                quit = key == 'q'
                key = window.getKey()

def visualizeDataset():
    with h5py.File(TRAIN_DATABASE_PATH, 'r') as infile:
        visualizer(infile['data'])

def visualize(visualizeNegatives=False, scale = SCALES[0][0]):
    fmt = 'face%d.hdf' if not visualizeNegatives else 'neg%d.hdf'
    db =  data.loadDatabase(fmt % scale)
    visualizer(db)