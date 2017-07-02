import h5py
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator

from data import DATASET_LABEL, RANDOM_SEED
from FaceDetection import DEBUG

DEFAULT_FIT_SAMPLE_SIZE = 100000

FLIP_PARAM_KEY = 'flip'

class ImageNormalizer():
    STANDARD_NORMALIZATION = 'std_norm'
    MIN_MAX_SCALING = 'min_max'
    ZCA_WHITENING = 'zca'

    FLIP_HORIZONTAL = 'fliph'
    FLIP_VERTICAL = 'flipv'
    FLIP_HORIZONTAL_AND_VERTICAL = 'flipvh'

    def __init__(self, posDatasetFilePath, negDatasetFilePath, normMethod = STANDARD_NORMALIZATION, sampleSize = DEFAULT_FIT_SAMPLE_SIZE, debug = DEBUG):
      self.X = None
      self.preprocessParams = {}
      self.imagePreprocessor = None
      self.noAugmentationImagePreprocesor = None
      self.normMethod = normMethod
      self.normalizationParams = {}

      for filePath in (posDatasetFilePath, negDatasetFilePath):
        with h5py.File(filePath, 'r') as inFile:
          dataset = inFile[DATASET_LABEL]
          if self.X is None: self.X = np.zeros((0, *dataset.shape[1:]), dtype = dataset[0].dtype)
          self.X = np.vstack((self.X, dataset[min(sampleSize, len(dataset)):]))

      if normMethod == ImageNormalizer.STANDARD_NORMALIZATION:
        self.preprocessParams['featurewise_center'] = True
        self.preprocessParams['featurewise_std_normalization'] = True
      elif normMethod == ImageNormalizer.MIN_MAX_SCALING:
        self.preprocessParams['rescale'] = 1./255
      elif normMethod == ImageNormalizer.ZCA_WHITENING:
        self.preprocessParams['zca_whitening'] = True

      self.normalizationParams.update(self.preprocessParams)

    def _isFitNessecary(self, normMethod):
      return normMethod in (ImageNormalizer.STANDARD_NORMALIZATION, ImageNormalizer.ZCA_WHITENING)

    # see https://keras.io/preprocessing/image/ for parameter names and available features
    def addDataAugmentationParams(self, params):
      if FLIP_PARAM_KEY in params.keys():
        flip_val = params.get(FLIP_PARAM_KEY)
        params['vertical_flip'] = flip_val == ImageNormalizer.FLIP_VERTICAL or flip_val == ImageNormalizer.FLIP_HORIZONTAL_AND_VERTICAL
        params['horizontal_flip'] = flip_val == ImageNormalizer.FLIP_HORIZONTAL or flip_val == ImageNormalizer.FLIP_HORIZONTAL_AND_VERTICAL
        del params[FLIP_PARAM_KEY]

      self.preprocessParams.update(params)

    def preprocess(self, images, labels = None, batchSize = None, shuffle = False, useDataAugmentation = True, seed = RANDOM_SEED):
      if self.imagePreprocessor is None:
        self.imagePreprocessor = ImageDataGenerator(**self.preprocessParams)
        self.noAugmentationImagePreprocesor = ImageDataGenerator(**self.normalizationParams)

        if self._isFitNessecary(self.normMethod):
          self.imagePreprocessor.fit(self.X)
          self.noAugmentationImagePreprocesor.fit(self.X)

          del self.X

      returnGenerator = batchSize is not None
      preprocessor = self.imagePreprocessor if useDataAugmentation else self.noAugmentationImagePreprocesor
      labels = np.zeros(len(images)) if labels is None else labels
      batchSize = len(images) if batchSize is None else batchSize

      generator = preprocessor.flow(images, labels, batch_size = batchSize, shuffle = shuffle, seed = seed)
      return generator if returnGenerator else next(generator)[0]
