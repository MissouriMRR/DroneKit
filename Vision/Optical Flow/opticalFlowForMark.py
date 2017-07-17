# Author(s): Christopher O'Toole
# Dependencies: OpenCV 3.2, pyrealsnese 2.0, numpy 1.13
# Execution instructions: Run with Python 3 interpreter

import logging
logging.basicConfig(level = logging.WARN)

import pyrealsense as pyrs
from pyrealsense import offline

import cv2
import numpy as np

#TODO: Replace with actual device serial number
DEVICE_SERIAL_NO = 1337
FPS = 60
INPUT_WIDTH = 320
INPUT_HEIGHT = 240
REGION_SIZE = 10

DEFAULT_OUTPUT_RATE = 15
DEFAULT_IMAGE_WIDTH = 64
DEFAULT_IMAGE_HEIGHT = 64
DEFAULT_NUMBER_OF_FEATURES = 20
DEFAULT_CONFIDENCE_MULTIPLIER = 1.645 

cam = None
depth_intrin = None
color_intrin = None
depth_scale = 0

class OpticalFlowOpenCV():
    def __init__(self, f_length_x, f_length_y, output_rate = DEFAULT_OUTPUT_RATE, img_width = DEFAULT_IMAGE_WIDTH, 
                 img_height = DEFAULT_IMAGE_HEIGHT, num_feat = DEFAULT_NUMBER_OF_FEATURES, conf_multi = DEFAULT_CONFIDENCE_MULTIPLIER):
        self.image_width = img_width
        self.image_height = img_height
        self.focal_length_x = f_length_x
        self.focal_length_y = f_length_y
        self.output_rate = output_rate
        self.num_features = num_feat
        self.confidence_multiplier = conf_multi
        
        self.initLimitRate()

    def initLimitRate(self):
        self.sum_flow_x = 0
        self.sum_flow_y = 0
        self.sum_flow_quality = 0
        self.valid_frame_count = 0

    def limitRate(self, flow_quality, frame_time_us, dt_us, flow_x, flow_y):
        if not hasattr(OpticalFlowOpenCV.limitRate, 'time_last_pub'):
            OpticalFlowOpenCV.limitRate.time_last_pub = 0
        
        if self.output_rate <= 0:
            dt_us[0] = frame_time_us - OpticalFlowOpenCV.limitRate.time_last_pub
            OpticalFlowOpenCV.limitRate.time_last_pub = frame_time_us
            return flow_quality

        if flow_quality > 0:
            self.sum_flow_x += flow_x[0]
            self.sum_flow_y += flow_y[0]
            self.sum_flow_quality += flow_quality
            self.valid_frame_count += 1

        if (frame_time_us - OpticalFlowOpenCV.limitRate.time_last_pub) > 1/self.output_rate:
            average_flow_quality = 0

            if self.valid_frame_count > 0:
                average_flow_quality = self.sum_flow_quality//self.valid_frame_count 
            
            flow_x[0] = self.sum_flow_x
            flow_y[0] = self.sum_flow_y

            self.initLimitRate()
            dt_us[0] = frame_time_us - OpticalFlowOpenCV.limitRate.time_last_pub
            OpticalFlowOpenCV.limitRate.time_last_pub = frame_time_us

            return average_flow_quality

    
try:
    stream_settings = {'fps': FPS, 'width': INPUT_WIDTH, 'height': INPUT_HEIGHT}
    color_stream, depth_stream = (pyrs.ColorStream(**stream_settings), pyrs.DepthStream(**stream_settings))
    cam = pyrs.Device(streams = [color_stream, depth_stream])
    depth_intrin = cam.depth_instrinsics
    depth_scale = cam.depth_scale
    color_intrin = cam.color_intrinsics

    flow_x = [0]
    flow_y = [0]
    last_time_stamp = 0

    def display_next_frame( ):
      color_image, depth_image = (cam.color, cam.depth)
      frame_time_stamp = cam.get_frame_timestamp(color_stream)
      global last_time_stamp
      global flow_x
      global flow_y

      center_pixel = np.array(list(depth_image.shape[:2])) // 2
      region_start = center_pixel - REGION_SIZE // 2
      region_end = center_pixel + REGION_SIZE // 2
      region_area = REGION_SIZE ** 2
      depth_image_area = np.prod(depth_image.shape[:2])
      X = (np.arange(depth_image_area) % depth_image.shape[1]).reshape(*depth_image.shape[:2])
      Y = (np.arange(depth_image_area) % depth_image.shape[0]).reshape(*depth_image.shape[-2::1]).transpose()
      
      average_depth = np.sum(((Y * depth_image + X)*depth_scale)[region_start[0]:region_end[0], region_start[1]:region_end[1]])/region_area
      delta = frame_time_stamp - last_time_stamp

      last_time_stamp = frame_time_stamp



finally:
    pyrs.stop()
    if cam is not None:
        cam.stop()