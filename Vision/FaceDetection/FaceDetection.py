from timeit import default_timer as timer

WINDOW_TITLE = 'Face Detector Test'
TEST = False

TRAIN = True
TRAIN_CALIB = False

LIVE_WINDOW_TITLE = 'RealSense Test'
LIVE = False

EVAL = False

PROFILE = True
DEBUG = True

STAGE_IDX = 1

GREEN = (0, 255, 0)
THICKNESS = 3

if __name__ == '__main__':
    import cv2
    import data
    from visualize import visualizer, cv2Window
    from detect import detectMultiscale

    def predictionCallback(img):
        start = timer()
        detections = detectMultiscale(img, STAGE_IDX)

        if PROFILE:
            print('Prediction took %fs' % (timer() - start,))
        
        for (xMin, yMin, xMax, yMax) in detections: 
            cv2.rectangle(img, (xMin, yMin), (xMax, yMax), GREEN, THICKNESS)


    if TEST:
        visualizer(data.getTestImagePaths(), predictionCallback, WINDOW_TITLE)
    elif TRAIN:
        from train import train
        train(STAGE_IDX, TRAIN_CALIB)
    elif LIVE:
        from RealSense import Streamer, LiveDisplay

        with cv2Window(LIVE_WINDOW_TITLE) as win, Streamer() as stream:
            liveStream = LiveDisplay(stream, win)
            liveStream.run(predictionCallback)
    elif EVAL:
        from eval import plot_precision_recall_vs_threshold
        plot_precision_recall_vs_threshold(STAGE_IDX, TRAIN_CALIB)
