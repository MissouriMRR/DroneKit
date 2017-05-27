import numpy as np

from annotation import RectangleAnnotation

def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate

def detections2boxes(detections):
    numDetections = len(detections)
    boxes = np.zeros((numDetections, 4))

    for i, detection in enumerate(detections):
        assert type(detection) is RectangleAnnotation
        boxes[i] = detection.coords
            
    return boxes

def annotations2matrix(func):
    def convert_param(detections, *args, **kwargs):
        if isinstance(detections, (tuple, list)):
            detections = detections2boxes(detections)
    
        return func(detections, *args, **kwargs)

    return convert_param