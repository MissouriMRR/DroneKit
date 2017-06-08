#!/usr/bin/env python3.5
WINDOW_TITLE = 'Face Detector Test'
TEST = False

TRAIN = True
TRAIN_CALIB = True

PROFILE = False
DEBUG = False

if __name__ == '__main__':
    import cv2
    import cProfile

    import data
    from visualize import cv2Window
    from data import SCALES

    STAGE_IDX = 0

    if TEST:
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
        train(STAGE_IDX, TRAIN_CALIB)
        