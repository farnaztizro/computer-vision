from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):

    return 1.0 / (1 + np.exp(-x))

def predict(X, W):

    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

# the diferrent between this and gradient descent is next_batch function

def next_batch(X, y, batchSize):

    for i in np.arange(0, X.shape[0], batchSize):
        yield(X[i:i + batchSize], y[i:i + batchSize])  
        
             