# Author(s): Christopher O'Toole
# Dependencies: OpenCV 3.2, pyrealsnese 2.0, numpy 1.13
# Execution instructions: Run with Python 3 interpreter

from __future__ import division
from sys import stdout

import pyrealsense as pyrs
import logging
import numpy as np


class RangeFinder():
    FPS = 60
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 480
    REGION_SIZE = 10

    def __init__(self):
        cam = None
        depth_scale = 0
        logging.basicConfig(level = logging.WARN)

    def initialize_camera(self):
        try:
            pyrs.start()
            self.cam = pyrs.Device(device_id=0, streams = [pyrs.stream.ColorStream(fps = 60, width=self.INPUT_WIDTH, height=self.INPUT_HEIGHT), pyrs.stream.DepthStream(fps = 60, width=self.INPUT_WIDTH, height=self.INPUT_HEIGHT)])
            self.depth_scale = self.cam.depth_scale
        except pyrs.utils.RealsenseError as ex:
            stdout.write("\nFailed to initialize RealSense.\n")
            stdout.write(ex)
            stdout.flush()

    def get_average_depth(self):
        self.cam.wait_for_frames()
        color_image, depth_image = (self.cam.color, self.cam.depth)

        center_pixel = np.array(list(depth_image.shape[:2])) // 2
        region_start = center_pixel - self.REGION_SIZE // 2
        region_end = center_pixel + self.REGION_SIZE // 2
        region_area = self.REGION_SIZE ** 2
        depth_image_area = np.prod(depth_image.shape[:2])
        depth_w, depth_h = (depth_image.shape[:2])
        X = (np.arange(depth_image_area) % depth_image.shape[1]).reshape(depth_w, depth_h)
        Y = (np.arange(depth_image_area) % depth_image.shape[0]).reshape(depth_h, depth_w).transpose()
        
        average_depth = np.sum(((Y * depth_image + X)*self.depth_scale)[region_start[0]:region_end[0], region_start[1]:region_end[1]])/region_area
        return average_depth

    def send_frame_to_ground(self):
        pass

    def shutdown(self):
        pyrs.stop()
        if self.cam is not None:
            self.cam.stop()
