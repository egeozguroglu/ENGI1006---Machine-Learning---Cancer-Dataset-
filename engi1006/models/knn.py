import numpy as np
from math import sqrt
from statistics import mode


def NNClassifier(training, testing, training_labels, testing_labels, k):
    '''Runs the Nearest Neighbor classifier:

    Args:
        training: the subset of data corresponding to the training data as a numpy matrix
        testing:  the subset of data corresponding to the testing data as a numpy matrix
        training_labels: the labels for the training data as a numpy array
        testing_labels: the labels for the testing data as a numpy array
        k: the number of nearest neighbors to use

    This function should do the following:

    - preallocate an array `labels` for the predicted labels of the testing data
    - for each row in the testing data, use knn to predict the label
    - at the end, return what percentagle of labels matched, i.e. how many labels in `labels` matched the label in `testing_labels`
    '''
    # preallocate labels
    labels = np.zeros(len(testing_labels))
    correct = 0

    # for each point
    for _ in range(len(testing_labels)):
        # run knn on each point and assign its label into labels
        predicted_labels = knn(training, training_labels, testing[_], k)
        real_label = testing_labels[_]
        
        if real_label == predicted_labels:
            correct += 1

    # return % where prediction matched actual
    return correct/len(testing_labels) 


def knn(data, data_labels, vector, k):
    '''knn should calculate the nearest neighbor

    data: the numpy array of training data
    data_labels: the numpy array of labels for the training data
    vector: a row from the testing data to calculate nearest neighbors
    k: how many nearest neighbors to find


    This function should find the `k` nearest rows in `data` relative to
    `vector`, and take a vote amongst their labels. Whichever has more (b or m), return
    that value'''
    # preallocate distance array
    distances = np.zeros(len(data_labels))

    # for each point in data
    for i in range(len(distances)):
        # calculate the distance to vector, store in distance array
        distances[i] = sqrt(((data[i].astype(float) - vector.astype(float)) ** 2).sum())
        
    # sort distances, and get indexes to use in data_labels (look at np.argsort)
    #indexes gives you the orderif you were to sort distances. 
    indexes = np.argsort(distances)

    # take vote amongs top labels
    to_vote = data_labels[indexes]
    return mode(to_vote[:k])

