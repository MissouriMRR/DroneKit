# Author(s): Christopher O'Toole
# Dependencies: OpenCV 3.2, pyrealsnese 2.0, numpy 1.13
# Execution instructions: Run with Python 3 interpreter

from __future__ import division

import logging
logging.basicConfig(level = logging.WARN)

import pyrealsense as pyrs

# import cv2
import numpy as np


FPS = 60
INPUT_WIDTH = 640
INPUT_HEIGHT = 480
REGION_SIZE = 10

cam = None
color_intrin = None
depth_scale = 0
    
try:
    pyrs.start()
    cam = pyrs.Device(device_id=0, streams = [pyrs.stream.ColorStream(fps = 60, width=INPUT_WIDTH, height=INPUT_HEIGHT), pyrs.stream.DepthStream(fps = 60, width=INPUT_WIDTH, height=INPUT_HEIGHT)])
    depth_scale = cam.depth_scale
    color_intrin = cam.color_intrinsics

    def get_average_depth( ):
      color_image, depth_image = (cam.color, cam.depth)

      center_pixel = np.array(list(depth_image.shape[:2])) // 2
      region_start = center_pixel - REGION_SIZE // 2
      region_end = center_pixel + REGION_SIZE // 2
      region_area = REGION_SIZE ** 2
      depth_image_area = np.prod(depth_image.shape[:2])
      depth_w, depth_h = (depth_image.shape[:2])
      X = (np.arange(depth_image_area) % depth_image.shape[1]).reshape(depth_w, depth_h)
      Y = (np.arange(depth_image_area) % depth_image.shape[0]).reshape(depth_h, depth_w).transpose()
      
      average_depth = np.sum(((Y * depth_image + X)*depth_scale)[region_start[0]:region_end[0], region_start[1]:region_end[1]])/region_area
      return average_depth

    while(True):
        cam.wait_for_frames()
        print('Average depth:', get_average_depth())
        # cv2.imshow('color', cam.color)
        # cv2.imshow('depth', cam.depth)
        # cv2.waitKey(1)

except KeyboardInterrupt:
    pyrs.stop()
    if cam is not None:
        cam.stop()
