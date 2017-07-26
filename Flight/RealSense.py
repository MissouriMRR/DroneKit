# Author(s): Christopher O'Toole
# Dependencies: OpenCV 3.2, pyrealsnese 2.0, numpy 1.13
# Execution instructions: Run with Python 3 interpreter

from __future__ import division
from sys import stdout
from time import sleep

import pyrealsense as pyrs
import logging
import numpy as np
import dronekit



class RangeFinder():
    FPS = 60
    INPUT_WIDTH = 320
    INPUT_HEIGHT = 240
    REGION_SIZE = 20
    MIN_REALSENSE_DISTANCE_CM = 30
    MAX_REALSENSE_DISTANCE_CM = 1000
    MAV_SENSOR_ROTATION_PITCH_270 = 25
    VEHICLE_CONNECTION_STRING = "127.0.0.1:14550"

    def __init__(self):
        cam = None
        depth_scale = 0
        logging.basicConfig(level = logging.WARN)
        self.color_image = None
        self.depth_image = None
        self.vehicle = None

    def initialize_camera(self):
        try:
            pyrs.start()
            self.cam = pyrs.Device(device_id=0, streams = [pyrs.stream.ColorStream(fps = 60, width=self.INPUT_WIDTH, height=self.INPUT_HEIGHT), pyrs.stream.DepthStream(fps = 60, width=self.INPUT_WIDTH, height=self.INPUT_HEIGHT)])
            self.depth_scale = self.cam.depth_scale
        except pyrs.utils.RealsenseError as ex:
            stdout.write("\nFailed to initialize RealSense.\n")
            stdout.write(str(ex))
            stdout.flush()
    
    def connect_to_vehicle(self):
        if(self.vehicle == None):
            self.vehicle = dronekit.connect(self.VEHICLE_CONNECTION_STRING)

    def get_frame(self):
        self.cam.wait_for_frames()
        self.color_image = self.cam.color
        self.depth_image = self.cam.depth

    def get_average_depth(self):
        self.get_frame()
        center_pixel = np.array(list(self.depth_image.shape[:2])) // 2
        region_start = center_pixel - self.REGION_SIZE // 2
        region_end = center_pixel + self.REGION_SIZE // 2
        region_area = self.REGION_SIZE ** 2
        depth_image_area = np.prod(self.depth_image.shape[:2])
        depth_w, depth_h = (self.depth_image.shape[:2])
        X = (np.arange(depth_image_area) % self.depth_image.shape[1]).reshape(depth_w, depth_h)
        Y = (np.arange(depth_image_area) % self.depth_image.shape[0]).reshape(depth_h, depth_w).transpose()
        
        average_depth = np.sum(((Y * self.depth_image + X)*self.depth_scale)[region_start[0]:region_end[0], region_start[1]:region_end[1]])/region_area
        return average_depth

    def send_frame_to_ground(self):
        pass
    
    def send_distance_message(self):
        try:
            while(True):
                distance = self.get_average_depth()

                message = self.vehicle.message_factory.distance_sensor_encode(
                    0,                                             # time since system boot, not used
                    self.MIN_REALSENSE_DISTANCE_CM,                # min distance cm
                    self.MAX_REALSENSE_DISTANCE_CM,                # max distance cm
                    distance,                                      # current distance, must be int
                    0,                                             # type = laser
                    0,                                             # onboard id, not used
                    self.MAV_SENSOR_ROTATION_PITCH_270,            # Downward facing range sensor.
                    0                                              # covariance, not used
                )
                self.vehicle.send_mavlink(message)
                self.vehicle.commands.upload()
                sleep(0.1)

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        pyrs.stop()
        if self.cam is not None:
            self.cam.stop()






