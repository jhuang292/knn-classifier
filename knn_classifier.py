#!/usr/bin/python3.6
import sys
import operator
import time
import random
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cdist

correct = 0
labels = []
confusion = {}
# num_features = 0
trainFeatures = []

def main(k, train, test):
    now = time.time()
    np.set_printoptions(threshold=np.inf)
    global num_features,confusion,trainFeatures,labels
    trainFeatures, trainSet = loadFile(train)
    testFeatures, testSet = loadFile(test)
    num_features = len(testFeatures) - 1
    labels = trainSet[:,-1]
    # trainSet[:,0:num_features] trainSet[:,0:-1])
    miu = np.array([i.mean() for y, i in enumerate(trainSet[:,0:-1].transpose())])
    sigma = np.array([i.std() for y, i in enumerate(trainSet[:,0:-1].transpose())])
    stdTrainSet = standardize(miu,sigma,trainSet)
    stdTestSet = standardize(miu,sigma,testSet)
    # stdTrainSet = np.array([[(i[x]-miu[x])/sigma[x] for x, j in enumerate(trainFeatures[0:-1])] for y, i in enumerate(trainSet[:,0:-1])])
    # np.concatenate((stdTrainSet, trainSet[:,-1].transpose()), axis=1)
    # np.concatenate((stdTestSet, testSet[:,-1].T), axis=1)
    confusion = {y: {x: 0 for x in labels} for y in labels}
    knn(k, stdTrainSet, stdTestSet)
    print("Time Elapsed: %d" % (time.time() - now))

def loadFile(file_name):
    with open(file_name) as f:
        data = json.load(f)
    listFeatures = np.array(data['metadata']['features'])
    listData = np.array(data['data'])
    return listFeatures, listData

def standardize(miu,sigma,set):
    global trainFeatures
    for y, i in enumerate(set[:,0:-1]):
        for x, j in enumerate(trainFeatures[0:-1]):
            if sigma[x] == 0:
                set[y][x] = (i[x]-miu[x])
            else:
                set[y][x] = (i[x]-miu[x])/sigma[x]
    return set

# Manhattan distance for numeric features and Hamming distance for categorical features.
def knn(k, train, test):
    global correct,num_features
    with open("knn_confusion.txt",'w') as f:
        for x in range(len(test)):
            label_neighbors = ""
            # distances = ManhattanDistances(train, test[x])
            distances = getDistances(train, test[x])
            distances_sorted = sorted(distances.items(), key=lambda kv: kv[1])#.items() returns tuple (k,v)
            # distances.sort(key=operator.itemgetter(1))#for 2-d array
            # np.sort(distances,axis=1)#for np array
            neighbors = [train[y] for y in [z[0] for z in distances_sorted[:int(k)]]]#split array
            response = getResponse(neighbors)
            for n in range(int(k)):
                # label_neighbors += str(neighbors[n][num_features])+","
                label_neighbors += str(neighbors[n][num_features])+","
            print("%s%d" % (label_neighbors,response))
            # sys.exit(0)
            confusionAccuracy(x, test[x][-1], response, f)
    # print_confusion()
    # print("Accuracy: %d %%" % (correct * 100 / len(test)))
    return correct/len(test)

def getDistances(train, test):
    global num_features, trainFeatures
    # dist = np.array([[i,0] for i in range(len(train))])
    dist = {}
    for idx_train,val_train in enumerate(train):
        rowDist = 0
        for idx_feature, val_feature in enumerate(trainFeatures[0:num_features,:]):
            if val_feature[1] == 'numeric':
                # dist += ManhattanDistances(val_train[idx_test],test[idx_test])
                rowDist += abs(val_train[idx_feature] - test[idx_feature])
            else:
                rowDist += HammingDistances(val_train[idx_feature],test[idx_feature])
        # dist[idx_train][1] = rowDist
        dist[idx_train] = rowDist
    return dist

# Euclidean Distance
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

# Hamming distances - count the number of features for which two instances differ
def HammingDistances(val1, val2):
    if val1 == val2:
        return 0
    return 1
# Manhattan distance
# def ManhattanDistances(val1, val2):
#     return abs(val1-val2)
def ManhattanDistances1(train, test):
    diffs = [[y, sum([abs(test[x] - j) for x, j in enumerate(i)])] for y, i in enumerate(train)]
    return diffs

def getResponse(neighbors):
    classifier = {} #dict
    for x in neighbors:
        if x[-1] in classifier:
            classifier[x[-1]] += 1
        else:
            classifier[x[-1]] = 1
    max_keys = max(classifier.keys(), key=(lambda k: classifier[k]))
    # max_value = max(classifier.values())  # maximum value
    # max_keys = [k for k, v in classifier.items() if v == max_value] # getting all keys containing the `maximum`
    return max_keys

def confusionAccuracy(n, actual, predicted, outfile_confusion):
    global correct,confusion
    confusion[actual][predicted] += 1
    if actual == predicted:
        correct += 1
    outfile_confusion.write("%d,%d %d\n" % (n,actual,predicted))



if __name__ == "__main__":
    # $ ./knn_classifier <INT k> <TRAINING SET> <TEST SET> > outfile.txt
    main(sys.argv[1], sys.argv[2], sys.argv[3])