import data
from visualize import cv2Window

import cv2
import cProfile

WINDOW_TITLE = 'Face Detector Test'
TEST = True

TRAIN = False
TRAIN_CALIB = False
TRAIN_CLASSIFIER = False
TRAIN_SCALE = 24

PROFILE = False

if __name__ == '__main__':
    if TEST:
        from detect import stage2_predict_multiscale
        
        with cv2Window( WINDOW_TITLE ) as window:
            annotations = data.getFaceAnnotations()
            posImgPaths = tuple(annotations.keys())
            i = 0
            key = ''
            
            while key != 'q':
                img = cv2.imread(posImgPaths[i])
                
                for detection in stage2_predict_multiscale(img):
                    detection.draw(img)

                if PROFILE:
                    cProfile.run('stage2_predict_multiscale(img)')
                window.show(img)
                key = window.getKey()
                
                while key not in 'npq':
                    key = window.getKey()
                
                if key == 'n':
                    i = ( i + 1 ) % len(posImgPaths)
                elif key == 'p':
                    i = i - 1 if i > 0 else len(posImgPaths)-1

    elif TRAIN:
        from train import train
        from detect import NET_FILE_NAMES, CALIB_NET_FILE_NAMES

        if TRAIN_CLASSIFIER:
            train(NET_FILE_NAMES.get(TRAIN_SCALE), TRAIN_SCALE, False)
        if TRAIN_CALIB:
            train(CALIB_NET_FILE_NAMES.get(TRAIN_SCALE), TRAIN_SCALE, False, True)
    