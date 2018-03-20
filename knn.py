from numpy import *
import operator
from os import listdir
import time
import pdb

def createDataSet():
	group = array([[1.0, 1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.1
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
			datingLabels[numTestVecs:m],3)
		print "the classifier came back with: %d, the real answer is: %d"\
		% (classifierResult, datingLabels[i])
		if (classifierResult != datingLabels[i]): errorCount += 1.0
	print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])

	return returnVect

def handwritingClassTest():
	start = time.clock()
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, \
			trainingMat, hwLabels, 3)
		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
		if (classifierResult != classNumStr): errorCount += 1.0
	print "\nthe total number of errors is:%d" % errorCount
	print "\nthe total error rate is: %f" % (errorCount/float(mTest))
	print "\nTime used: %f" % (time.clock()-start)

class node:
	def __init__(self):
		self.data = []
		self.parent = None
		self.right = None
		self.left = None
		self.split = []
		self.median = []
		self.leaf = True

def buildKDTree(dataMat):

	if len(dataMat) < 1:
		return 

	root = node()
	
	variance = dataMat.var(axis=0)
	root.split = variance.argsort()[-1]
	#pdb.set_trace()
	dump = dataMat[:,root.split].argsort()
	n = len(dataMat)
	#medianIdx = dump[int((n-1)/2)]
	medianIdx = dump[int(n/2)]
	root.data = dataMat[medianIdx,:]
	root.median = dataMat[medianIdx,root.split]

	leftIdx = dataMat[:,root.split] < root.median
	leftData = dataMat[leftIdx,:]
	root.left = buildKDTree(leftData)
	if root.left is not None:
		root.leaf = False
		root.left.parent = root

	rightIdx = dataMat[:,root.split] > root.median
	rightData = dataMat[rightIdx,:]
	root.right = buildKDTree(rightData)
	if root.right is not None:
		root.leaf = False
		root.right.parent = root

	return root

def searchKDTree(x, root, nearestNode = None, nearestDist = float('inf')):
	currNode = root
	visitedList = []
	visitedList.append(currNode)

	while currNode.leaf == False:
		#pdb.set_trace()
		if currNode.median > x[currNode.split]:
			currNode = currNode.left
			#visitedList.append(currNode)
		else:
			currNode = currNode.right
		visitedList.append(currNode)

	while len(visitedList) != 0:
		pdb.set_trace()
		currNode = visitedList.pop()
		dist = sqrt(sum((currNode.data - x)**2))
		if dist < nearestDist:
			nearestNode = currNode
			nearestDist = dist

		if len(visitedList) != 0:
			if abs(x[currNode.parent.split] - currNode.parent.median) < nearestDist:
				if currNode == currNode.parent.left:
					nearestNode,nearestDist = searchKDTree(x, currNode.parent.right, nearestNode, nearestDist)
				else:
					nearestNode,nearestDist = searchKDTree(x, currNode.parent.left, nearestNode, nearestDist)

	return nearestNode, nearestDist




