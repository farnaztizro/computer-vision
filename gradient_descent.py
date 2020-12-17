from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):

    return 1.0 / (1 + np.exp(-x))

def predict(x, W):

    preds = sigmoid_activation(x.dot(W))
    # apply a step function to threshold the outputs to binary class label
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")   
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args)

# generate some data to classify
# n_sample: datapoints
# n_features: dimention, if 1 =linear, default:2
# centers: number of classes, if is None =3
# cluster_std:enheraf meyar
# shuffle: qareqati, default=true
# random_state: ,default:None
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1’s as the last entry in the feature matrix
X =np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

