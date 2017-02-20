import cv2
import numpy as np
import os, sys, contextlib
import winsound, win32api, win32con

NEGATIVE_IMAGES_FOLDER = r'C:\Users\Christopher\Downloads\neg'
POSITIVE_IMAGES_FOLDER = r'C:\Users\Christopher\Downloads\pos'
SHUTTER_SOUND = r'C:\Users\Christopher\Downloads\shutter.wav'
SCALE_TO = 400

PENCIL_LINE_THRESHOLD = 150
MORPHOLOGY_KERNEL_SIZE = 2

# rotate an image counter-clockwise
# credit goes to: http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

def makePencilUpright( imagePath ):
    pencil = cv2.imread( imagePath )
    edges = cv2.Canny( pencil, 100, 275, 5 )
    lines = cv2.HoughLines( edges, 1, np.pi / 720., PENCIL_LINE_THRESHOLD )
    degrees = 0

    for rho, theta in lines[0]:
        unitVec = np.array([ np.cos(theta), np.sin( theta )])
        mag = 5
        p0 = unitVec * rho
        
        unitVec = np.array([ -unitVec[1], unitVec[0] ])
        p1 = p0 + mag * unitVec
        p2 = p0 - mag * unitVec

        lineVec = np.array([ p2[1] - p1[1], p2[0] - p1[0] ])
        lineVec /= np.linalg.norm( lineVec, ord = 1 )
        upVec = np.array([ 1, 0 ])
        degrees = np.degrees( np.arccos( upVec.dot( lineVec ) ) )

    pencil = rotate_bound( pencil, -degrees )
    cv2.imshow( 'pencil', pencil )
    cv2.waitKey( 0 )

def listdirpaths( dir ):
    for fileName in os.listdir( dir ):
        yield '%s\\%s' % ( dir, fileName )

def preprocessImages( folder, scaleTo = None, equalizeHistogram = True, convertToGrayscale = True ):
    for filePath in listdirpaths( folder ):
        image = cv2.imread( filePath, cv2.IMREAD_GRAYSCALE if convertToGrayscale else None )
        
        if equalizeHistogram:
            image = cv2.equalizeHist( image )
        if scaleTo:
            scaleFactor = scaleTo / float( max( image.shape[:2] ) )
            image = cv2.resize( image, None, fx = scaleFactor, fy = scaleFactor )

        cv2.imwrite( filePath, image )


def listFolder( folder, outputFileName, absPath = False ):
    outputFileName = '{}\\{}'.format( os.path.dirname( folder ), outputFileName )

    with open( outputFileName, 'wb' ) as out:
        for fileName in os.listdir( folder ):
                path = '{}\\{}'.format( folder, fileName ) if absPath else '{}/{}'.format( folder[folder.rfind( '\\' ) + 1:], fileName )
                out.write( ( path + os.linesep ).encode( ) )

def simplifyFileNamesInFolder( folder ):
    for i, path in enumerate( listdirpaths( folder ) ):
        os.rename( path, '{}\\{}{}'.format( folder, i, os.path.splitext( path )[1] ) )

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

    def takePicture( self, img, dir, fileName, sfx = None ):
        imgPath = '{}\\{}'.format( dir, fileName )
        cv2.imwrite( imgPath, img )

        if sfx:
            winsound.PlaySound( sfx, winsound.SND_FILENAME )

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


def takePicsWithWebcam( dir ):
    WINDOW_TITLE = 'Press space to take a picture!' 
    dirFileNames = os.listdir( dir )
    dirFileNames.extend( ( '1.png', ) )
    fileNameIndex = max( int( os.path.splitext( i )[0] ) for i in dirFileNames )

    if ( os.path.isdir( dir ) ):
        with WebcamCapture( ) as webcam, cv2Window( WINDOW_TITLE ) as window:
            for status, frame in webcam:
                if status:
                    if not window.show( frame ):
                        break
                    if win32api.GetAsyncKeyState( win32con.VK_SPACE ):
                        webcam.takePicture( frame, dir, '{}.png'.format( fileNameIndex ), SHUTTER_SOUND )
                        fileNameIndex += 1
    else:
        print( 'The directory {} is invalid!'.format( dir ), file = sys.stderr )

listFolder( NEGATIVE_IMAGES_FOLDER, 'negatives.txt' )