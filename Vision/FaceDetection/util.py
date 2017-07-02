import inspect 
import h5py
import tempfile
import os
import sys
import random

RAND_LOW, RAND_HIGH = (0, sys.maxsize)

def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate

def getDefaultParams(func):
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(args[len(args)-len(defaults):], defaults))

class TempH5pyFile(h5py.File):
	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
		self.filePath = self._generateFilePath()

		while os.path.isfile(self.filePath):
			self.filePath = self._generateFilePath()

	def _generateFilePath(self):
		return os.path.join(tempfile.gettempdir(), str(random.randint(RAND_LOW, RAND_HIGH)) + '.hdf')

	def getPath(self):
		return self.filePath

	def __enter__(self):
		super().__init__(self.filePath, *self.args, **self.kwargs)
		return self

	def close(self):
		try:
			super().close()
		finally: os.remove(self.filePath)

	def __exit__(self, *args):
		self.close()