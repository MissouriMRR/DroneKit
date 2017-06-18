import pyrealsense as pyrs
import pyrealsense.constants
import cv2

from collections import namedtuple
from timeit import default_timer as timer

AVAILABLE_COLOR_RESOLUTIONS_FOR_R200 = ((320, 240), (640, 480), (1280, 720), (1920, 1080))
AVAILABLE_DEPTH_RESOLUTIONS_FOR_R200 = ((320, 240), (480, 360), (628, 468))

DEFAULT_COLOR_FPS = 60
DEFAULT_COLOR_RES = AVAILABLE_COLOR_RESOLUTIONS_FOR_R200[1]

RS_FORMAT_FLAG_PREFIX = 'RS_FORMAT_'

_stream_flags = namedtuple('flags', ['COLOR_STREAM'])
_formats = {k.replace(RS_FORMAT_FLAG_PREFIX, ''): v for k, v in pyrealsense.constants.rs_format.__dict__.items() if k.startswith(RS_FORMAT_FLAG_PREFIX)}
_color_conversions = {_formats['BGR8']: None,
                      _formats['BGRA8']: cv2.COLOR_BGRA2BGR, 
                      _formats['RGB8']: cv2.COLOR_RGB2BGR, 
                      _formats['RGBA8']: cv2.COLOR_RGBA2BGR,
                      _formats['YUYV']: cv2.COLOR_YUV2BGR_YUYV}
STREAM_FLAGS = _stream_flags(COLOR_STREAM = 1)
_stream_names = {STREAM_FLAGS.COLOR_STREAM: 'color'}

class Service():
	def __enter__(self):
		pyrs.start()
		return self

	def __exit__(self):
		pyrs.stop()


class Device():
	def __init__(self, params):
		self.params = params

	def __enter__(self):
		self.device = pyrs.core.Device(**self.params)
		return self.device

	def __exit__(self):
		self.device.stop()

class Streamer():
	def __init__(self, flag = STREAM_FLAGS.COLOR_STREAM, res = DEFAULT_COLOR_RES, fps = DEFAULT_COLOR_FPS):
		self.flag = flag
		self.res = res
		self.fps = fps
		self.stream_params = {'width': self.res[0], 'height': self.res[1], 'fps': self.fps}
		self.possible_streams = {STREAM_FLAGS.COLOR_STREAM: pyrs.stream.ColorStream}

	def __enter__(self):
		self.service = Service()
		self.service.__enter__()
		self.stream = self.possible_streams.get(self.flag)(**self.stream_params)
		self.device = Device({'streams': [self.stream]})
		self.cam = self.device.__enter__()
		return self

	def next(self, block = True):
		if block:
			self.cam.wait_for_frames()
		else:
			if not self.poll_for_frame():
				return None

		colorConversionCode = _color_conversions.get(self.stream.format)
		img = getattr(self.cam, _stream_names.get(self.flag))

		if colorConversionCode:
			img = cv2.cvtColor(img, colorConversionCode)

		return img


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

			if callback is not None: 
				ret = callback(img)
				if ret is not None and type(ret) is type(img):
					img = ret
			
			self.window.show(img)

			frames += 1

			if self.window.getKey() == keyToQuit: 
				quit = True

			if (timer() - start >= 1):
				self.updateFPS(frames)
				start = timer()
				frames = 0