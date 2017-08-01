from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pyrealsense as pyrs
import pyrealsense.constants
import cv2

from collections import namedtuple
from timeit import default_timer as timer

AVAILABLE_COLOR_RESOLUTIONS_FOR_R200 = ((320, 240), (640, 480), (1280, 720), (1920, 1080))
AVAILABLE_DEPTH_RESOLUTIONS_FOR_R200 = ((320, 240), (480, 360), (628, 468))

REGION_SIZE = 50

DEFAULT_COLOR_FPS = 60
DEFAULT_COLOR_RES = AVAILABLE_COLOR_RESOLUTIONS_FOR_R200[1]

RS_FORMAT_FLAG_PREFIX = 'RS_FORMAT_'

_stream_flags = namedtuple('flags', ['COLOR_STREAM', 'DEPTH_STREAM'])
_formats = {k.replace(RS_FORMAT_FLAG_PREFIX, ''): v for k, v in pyrealsense.constants.rs_format.__dict__.items() if k.startswith(RS_FORMAT_FLAG_PREFIX)}
_color_conversions = {_formats['BGR8']: None,
                      _formats['BGRA8']: cv2.COLOR_BGRA2BGR, 
                      _formats['RGB8']: cv2.COLOR_RGB2BGR, 
                      _formats['RGBA8']: cv2.COLOR_RGBA2BGR,
                      _formats['YUYV']: cv2.COLOR_YUV2BGR_YUYV}

STREAM_FLAGS = _stream_flags(COLOR_STREAM = 1, DEPTH_STREAM = 2)
_stream_names = {STREAM_FLAGS.COLOR_STREAM: 'color'}

class Service():
  def __enter__(self):
    self.service = pyrs.Service()
    return self

  def getService(self):
    return self.service

  def __exit__(self):
    self.service.stop()

class Device():
  def __init__(self, params, service):
    self.params = params
    self.service = service

  def __enter__(self):
    self.device = self.service.Device(**self.params)
    return self.device

  def __exit__(self):
    self.device.stop()

class Streamer():
  def __init__(self, flag = STREAM_FLAGS.COLOR_STREAM, res = DEFAULT_COLOR_RES, fps = DEFAULT_COLOR_FPS):
    self.flag = flag
    self.res = res
    self.fps = fps
    self.average_depth = 0
    self.stream_params = {'width': self.res[0], 'height': self.res[1], 'fps': self.fps}

  def __enter__(self):
    self.service = Service()
    self.service.__enter__()
    self.streams = [pyrs.stream.ColorStream, pyrs.stream.DepthStream]
    self.device = Device({'streams': streams}, self.service.getService())
    self.cam = self.device.__enter__()
    return self   

  def get_average_depth(self, depth_image):
    return self.average_depth
 
  def next(self, block = True):
    if block:
      self.cam.wait_for_frames()
    else:
      if not self.cam.poll_for_frame():
        return None

    colorConversionCode = _color_conversions.get(self.streams[0].format)
    img = self.cam.color

    if colorConversionCode:
      img = cv2.cvtColor(img, colorConversionCode)

    center_pixel = np.array(list(self.cam.depth.shape[:2])) // 2
    region_start = center_pixel - self.REGION_SIZE // 2
    region_end = center_pixel + self.REGION_SIZE // 2
    region_area = self.REGION_SIZE ** 2
    depth_image_area = np.prod(self.cam.depth.shape[:2])
    depth_w, depth_h = (depth_image.shape[:2])
    X = (np.arange(depth_image_area) % self.cam.depth.shape[1]).reshape(depth_w, depth_h)
    Y = (np.arange(depth_image_area) % self.cam.depth.shape[0]).reshape(depth_h, depth_w).transpose()
    
    average_depth = np.sum(((Y * self.cam.depth + X)*self.depth_scale)[region_start[0]:region_end[0], region_start[1]:region_end[1]])/region_area

    self.average_depth = average_depth

    return img, self.cam.depth


  def __exit__(self, *args):
    self.device.__exit__()
    self.service.__exit__()

class LiveDisplay():
  def __init__(self, stream, window, showFPS = True):
    self.stream = stream
    self.window = window
    self.showFPS = showFPS
    self.originalWindowTitle = window.getTitle()

  def updateFPS(self, fps):
    self.window.setTitle('%s (%d FPS)' % (self.originalWindowTitle, fps))

  def run(self, callback = None, keyToQuit = 'q'):
    quit = False
    start = timer()
    frames = 0

    while not quit:
      img = self.stream.next()
      if callback is not None: callback(img)
      self.window.show(img)

      frames += 1

      if self.window.getKey() == keyToQuit: 
        quit = True

      if (timer() - start >= 1):
        self.updateFPS(frames)
        start = timer()
        frames = 0