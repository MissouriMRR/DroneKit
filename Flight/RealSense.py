# Author(s): Christopher O'Toole, Tanner Winkelman
# Dependencies: OpenCV 3.2, pyrealsnese 2.0, numpy 1.13
# Execution instructions: Run with Python 3 interpreter

from __future__ import division
from sys import stdout
from matplotlib import pyplot as plt

import pyrealsense as pyrs
import logging
import numpy as np
import socket
import json
import pickle

# better change this value; I don't know what address to put here
GROUND_IP = 'localhost'

# needs to be the same on the ground (server) program
PORT_NUMBER = 8089


import socket
import numpy as np
from cStringIO import StringIO

class numpysocket():
    def __init__(self):
        pass

    @staticmethod
    def startClient(server_address,image):
        if not isinstance(image,np.ndarray):
            print 'not a valid numpy image'
            return
        client_socket=socket.socket()
        port=PORT_NUMBER
        try:
            client_socket.connect((server_address, port))
            print 'Connected to %s on port %s' % (server_address, port)
        except socket.error,e:
            print 'Connection to %s on port %s failed: %s' % (server_address, port, e)
            return
        f = StringIO()
        np.savez_compressed(f,frame=image)
        f.seek(0)
        out = f.read()
        client_socket.sendall(out)
        client_socket.shutdown(1)
        client_socket.close()
        print 'image sent'
        pass


class RangeFinder():
    FPS = 60
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 480
    REGION_SIZE = 10
    client_socket = socket.socket( )

    def __init__(self):
        self.cam = None
        self.depth_scale = 0
        #self.logging.basicConfig(level = logging.WARN)
        # added by Tanner
        #self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.client_socket.connect((GROUND_IP, PORT_NUMBER))


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

    # added by Tanner
    def send_frame_to_ground(self):
        #serialized = pickle.dumps(self.cam.color, protocol=pickle.HIGHEST_PROTOCOL) # protocol 0 is printable ASCII
        #size = len( serialized )
        #self.client_socket.send( str(size) )
        #self.client_socket.send(serialized)
        self.get_average_depth()
        numpysocket.startClient(GROUND_IP,self.cam.color)
        plt.imshow(self.cam.color, interpolation='nearest')
        plt.show()

    def shutdown(self):
        pyrs.stop()
        if self.cam is not None:
            self.cam.stop()
