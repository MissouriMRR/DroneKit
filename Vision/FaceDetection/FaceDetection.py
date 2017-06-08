#!/usr/bin/env python3.5
import cProfile

WINDOW_TITLE = 'Face Detector Test'
TEST = True

TRAIN = False
TRAIN_CALIB = False

PROFILE = False
DEBUG = False

STAGE_IDX = 0

if __name__ == '__main__':
    import data
    from visualize import visualizer

    if TEST:
        visualizer(data.getTestImagePaths())
    elif TRAIN:
        from train import train
        train(STAGE_IDX, TRAIN_CALIB)
        