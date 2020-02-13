from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()

testIndex = [0, 50, 100]
#training data

trainData = np.delete(iris.data, testIndex, axis = 0)
trainTarget = np.delete(iris.target, testIndex)

#testing data
testTarget = iris.target[testIndex]
testData = iris.data[testIndex]

clf = tree.DecisionTreeClassifier()
clf.fit(trainData, trainTarget)

print(clf.predict(testData))
