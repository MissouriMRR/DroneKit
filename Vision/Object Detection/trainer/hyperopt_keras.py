from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import sys
import os
import numpy as np
import atexit

from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, space_eval

DEFAULT_NUM_FOLDS = 3
DEFAULT_NUM_EPOCHS = 50
DEFAULT_NUM_EVALS = 30

WEIGHTS_FILE_NAME = 'tune.hdf'

class HyperoptWrapper():
	def __init__(self):
		self.labelIdx = 0

	def _getParam(self, func, *args):
		param = func(str(self.labelIdx), *args)
		self.labelIdx += 1
		return param

	def uniform(self, low, high):
		return self._getParam(hp.uniform, low, high)

	def choice(self, *args):
		return self._getParam(hp.choice, list(args))

	def randint(self, low, high):
		return low + self._getParam(hp.randint, max(low, high - low))

	def loguniform(self, low, high):
		return self._getParam(hp.loguniform, np.log(low), np.log(high))

def parseParams(params):
	from .model import DROPOUT_PARAM_ID, OPTIMIZER_PARAMS, NORMALIZATION_PARAMS, TRAIN_PARAMS
	
	normalizationParams = {}
	compileParams = {}
	trainParams = {}
	dropouts = []

	for k, v in params.items():
		if k in NORMALIZATION_PARAMS:
			normalizationParams[k] = v
		elif k in OPTIMIZER_PARAMS:
			compileParams[k] = v
		elif k in TRAIN_PARAMS:
			trainParams[k] = v
		elif type(k) is str and k.startswith(DROPOUT_PARAM_ID):
			if not dropouts:
				dropouts.append(v)
			else:
				idx = int(k.replace(DROPOUT_PARAM_ID, ''))
				dropouts.insert(idx, v)

	return normalizationParams, compileParams, trainParams, dropouts

def optimize(func):
	def decorate(paramSpace, *args, **kwargs):		
		numEvals = kwargs.get('numEvals') or DEFAULT_NUM_EVALS
		if 'numEvals' in kwargs.keys(): del kwargs['numEvals']

		trials = Trials()
		callback = lambda params: func(params, *args, **kwargs)
		atexit.register(os.remove, WEIGHTS_FILE_NAME)
		best = fmin(callback, paramSpace, algo = tpe.suggest, max_evals = numEvals, trials = trials)
		return best, trials

	return decorate

@optimize
def tune(params, model, datasetManager, labels, metric, numFolds = DEFAULT_NUM_FOLDS, numEpochs = DEFAULT_NUM_EPOCHS, verbose = True):
	from .dataset import ClassifierDataset
	from .preprocess import ImageNormalizer

	log = lambda *args: print(*args) if verbose else None

	normalizationParams, compileParams, trainParams, dropouts = parseParams(params)
	batchSize = trainParams['batchSize']
	normMethod = normalizationParams['norm']
	del normalizationParams['norm']

	paths = datasetManager.getPaths()

	with ClassifierDataset(paths[0], paths[1], labels) as dataset:
		scores = []

		normalizer = ImageNormalizer(datasetManager.getPosDatasetFilePath(), datasetManager.getNegDatasetFilePath(), normMethod)
		normalizer.addDataAugmentationParams(normalizationParams)
		fitParams = {'compileParams': compileParams, 'dropouts': dropouts, 'batchSize': batchSize, 'verbose': False, 'saveFilePath': WEIGHTS_FILE_NAME}

		for i, (X_train, X_test, y_train, y_test) in enumerate(dataset.stratifiedSplitter(numFolds)):
			model.fit(X_train, X_test, y_train, y_test, datasetManager, normalizer, numEpochs, **fitParams)

			if not i: log('Using params:', params)
			metricParams = {'average': 'binary' if np.amax(y_test) == 1 else 'macro'}
			scores.append(model.eval(X_test, y_test, normalizer, metric, weightsFilePath = WEIGHTS_FILE_NAME, datasetManager = datasetManager, **metricParams))
			log('Fold %d score: %.5f' % (i, scores[-1]))

		metricAvg = np.mean(np.array(scores), dtype = np.float64)
		lossVariance = np.var(np.array(scores), dtype = np.float64)
		log('Average score: %.5f' % (metricAvg,))
		log('Variance: %.5f' % (lossVariance,))

	sys.stdout.flush()

	return {'loss': -metricAvg, 'loss_variance': lossVariance, 'status': STATUS_OK}

def getBestParams(paramSpace, best):
	return space_eval(paramSpace, {k: v[0] if type(v) is list else v for k, v in best.items()})