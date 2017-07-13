from __future__ import absolute_import, division, print_function, unicode_literals
from optparse import OptionParser
from timeit import default_timer as timer

WINDOW_TITLE = 'Face Detector Test'
LIVE_WINDOW_TITLE = 'RealSense Test'

PROFILE = True
DEBUG = False

GREEN = (0, 255, 0)
THICKNESS = 3

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--stage', dest='stageIdx', help='Cascade stage index', metavar = '[0-2]', default = 2)
    parser.add_option('-c', '--calib', action='store_true', dest = 'trainCalib', help='Use to train the calibration net for the given stage', default = False)
    parser.add_option('-v', '--visualize',  action='store_true', dest = 'testMode', help='View output of detector on different test images', default = False)
    parser.add_option('-l', '--live',  action='store_true', dest = 'liveMode', help='Test detector on a live Intel RealSense Stream', default = False)
    parser.add_option('-e', '--eval', action='store_true', dest='evalMode', help="Plot preicison+recall vs. threshold for given stage", default = False)
    parser.add_option('-t', '--train', action='store_true', dest='trainMode', help='Train either the classifier or calibrator for the given stage', default = False)

    (options, args) = parser.parse_args()

    def predictionCallback(img):
        import cv2
        from .detect import detectMultiscale
        start = timer()
        detections = detectMultiscale(img, int(options.stageIdx))

        if PROFILE:
            print('Prediction took %fs' % (timer() - start,))
        
        for (xMin, yMin, xMax, yMax) in detections: 
            cv2.rectangle(img, (xMin, yMin), (xMax, yMax), GREEN, THICKNESS)

    if options.testMode:
        from .visualize import visualizer
        import data
        visualizer(data.getTestImagePaths(), predictionCallback, WINDOW_TITLE)
    elif options.trainMode:
        from .train import train
        train(int(options.stageIdx), options.trainCalib)
    elif options.liveMode:
        from visualize import cv2Window
        from RealSense import Streamer, LiveDisplay

        with cv2Window(LIVE_WINDOW_TITLE) as win, Streamer() as stream:
            liveStream = LiveDisplay(stream, win)
            liveStream.run(predictionCallback)
    elif options.evalMode:
        from eval import plot_precision_recall_vs_threshold
        plot_precision_recall_vs_threshold(int(options.stageIdx), options.trainCalib)
