import numpy as np

def loadSimpData():
	dataMat = np.mat([[1,2.1],
					[2, 1.1],
					[1.3, 1],
					[1, 1],
					[2, 1]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return dataMat,classLabels

def stumpyClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray