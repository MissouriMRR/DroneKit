from collections import namedtuple
from sys import exit
from datetime import datetime
import os
import builtins
import cv2

WORKING_DIR = r'/home/christopher/Downloads/Roomba Dataset/Positives'

RESIZE_DOC = \
'''Syntax: resize <scaleBy>
  Resizes all images in the current working directory by the value scaleBy. 
  Example: 'resize .5' scales all images in the current directory to .5x their original width and height'''
ANNOTATE_DOC = \
'''Opens a dialog for annotating images with rectangular bounds.'''
EXTRACT_VIDEOS_DOC = \
'''Syntax: extractVideos <numFramesToSkip>
  Unpacks each video in the current working directory into a sqeuence of image files with numFramesToSkip frames skipped between each image'''

SUPPORTED_VIDEO_FILE_EXTENSIONS = ('.mp4', '.ogg')

def workingdir( func ):
    def wrapper( *args, **kwargs ):
        cwd = os.getcwd( )
        os.chdir( WORKING_DIR )
        res = func( *args, **kwargs )
        os.chdir( cwd )
        return res

    return wrapper

def help( command ):
    if command in Commands:
        for line in Commands[command].help.split( '\n' ):
            print( ' ', line )

        print()
    else:
        return False, 'help: Unknown command %s' % repr( command )

def commands( ):
    print()

    for commandName in Commands.keys( ):
        print( commandName )
        help( commandName )


@workingdir
def resize( scaleBy ):
    if ( scaleBy > 0 ):
        for fileName in os.listdir( '.' ):
            try:
                img = cv2.imread( fileName )
                resized = cv2.resize( img, (0, 0), fx = scaleBy, fy = scaleBy )
                cv2.imwrite( fileName, resized )
            except SystemError as e:
                print( 'resize: Could not resize image %s.' % repr( fileName ) )
    else:
        return False, 'resize: Scale parameter must be non-negative!' 

@workingdir
def extractVideos( numFramesToSkip ):
    cap = None
    
    if numFramesToSkip >= 0:
        for fileName in os.listdir('.'):
            if os.path.splitext(fileName)[1] in SUPPORTED_VIDEO_FILE_EXTENSIONS:
                try:
                    cap = cv2.VideoCapture(fileName)
                    ret = True
                    i = 0

                    while (cap.isOpened() and ret):
                        ret, frame = cap.read()
                        if not ret: continue
                        
                        if i < numFramesToSkip:
                            i += 1
                            continue
                        elif i == numFramesToSkip:
                            i = 0
                        
                        cv2.imwrite('%s%s' % (datetime.now(), '.jpg'), frame)

                    os.remove(fileName)
                except Exception as e:
                    return False, 'extractVideos: ' + str( e )
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
    else:
        return False, 'extractVideos: numFramesToSkip must be non-negative!'

def annotate( ):
    from Annotator import Annotator

    try:
        annotator = Annotator()
        annotator.loop()
    except AssertionError as e:
        return False, 'annotate: ' + str( e )

def quit( ):
    exit( 0 )

Command = namedtuple( 'Command', ['callback', 'help', 'paramTypes'] )
Commands = {
    'help': Command( callback = help, help = 'Displays usage info for a command', paramTypes = ( 'str', ) ),
    'resize' : Command( callback = resize, help = RESIZE_DOC, paramTypes = ( 'float', ) ),
    'extractvideos': Command( callback = extractVideos, help = EXTRACT_VIDEOS_DOC, paramTypes = ( 'int', ) ),
    'annotate' : Command( callback = annotate, help = ANNOTATE_DOC, paramTypes = tuple() ),
    'quit' : Command( callback = quit, help = 'Quits the program.', paramTypes = tuple() ),
    'commands' : Command( callback = commands, help = 'Displays all commands and their documentation', paramTypes = tuple() ),
}

def handleCommand( input ):
    words = input.split()
    commandName = words[0].lower() if words else None
    command = None
    params = []
    result = None

    if commandName in Commands:
        command = Commands[commandName]

        if len( command.paramTypes ) == len( words ) - 1:
            try:
                if ( command.paramTypes ):
                    for i in range(1, len(words)):
                        params.insert( -1, getattr( builtins, command.paramTypes[i - 1] )( words[i] ) )
            except ValueError as e:
                return False, '%s' % ( e, )

            result = command.callback( *params )

            if result:
                return result

        else:
            return False, '%s: Expected %d parameters, got %d' % ( commandName, len( command.paramTypes ), len( params ) )

    return True, ''

if __name__ == '__main__':
    assert os.path.isdir( WORKING_DIR ), 'WORKING_DIR constant is not a valid directory'

    print( 'Image Dataset Utility v1.0\n' \
           'Current Working Directory: %s\n' % ( WORKING_DIR ) )

    while True:
        status, error = handleCommand( input( ) )
        if not status: print( error )