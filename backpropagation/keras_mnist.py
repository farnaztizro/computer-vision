# encode our integer labels as vector labels
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# our network will be feedforward and layers will be added
from keras.models import Sequential
# implementation of our fully-connected layers
from keras.layers.core import Dense
# for our network to actually learn
# to optimize the parameters
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the full MNIST dataset
print("[INFO] loading MNIST (full) dataset...")
# load the dataset from disk
dataset = datasets.fetch_openml("MNIST Original")
# data normalizing --> scaling the pixel intensities to the range [0,1]
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

# encode our labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define our network architecture(784-256-128-10)
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
# only learn 10 weights(0-9)
model.add(Dense(10, activation="softmax"))

# train our network

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
