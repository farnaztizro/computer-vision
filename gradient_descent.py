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

#  handles randomly initializing our weight matrix 
print("[INFO] training...")
# sotune xo mirize tu satre in = 3x1
W = np.random.randn(X.shape[1], 1)
# print(X) = 3072x3
# initializes a list to keep track of our losses after each epoch
losses = []

# actual training and gradient descent

# start looping over the supplied number of --epochs
for epoch in np.arange(0, args["epochs"]):
    # giving us our predictions on the dataset
    # takes the dot product between our entire training set trainX and our weight matrix
    preds = sigmoid_activation(trainX.dot(W))

    # now we have our prediction, we need to determine the error which is the 
    # difference between our predictions and the true value
    # trainY = real value
    error = preds - trainY
    # compute the least squares error over our predictions
    # zigma formul
    loss = np.sum(error ** 2)
    losses.append(loss)

    # Now that we have our error, we can compute the
    # gradient and then use it to update our weight matrix W

    gradient = trainX.T.dot(error)

    # ta vaqti gradient b chizi k mikhaim berese bayad - bezarim ta berese b oni k mikhaim
    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


# classifier is now trained
# The next step is evaluation

print("[INFO] evaluating...")
preds = predict (trainX, W)
print(classification_report(testY, preds))

# handles plotting
# 1. visualize the dataset we are trying to classify
# 2. our loss over time

plt.style.use("ggplot")
# chanta chiz ru y nemudar(1bodi bashe nemikhad)
plt.figure()
plt.title("Data")
# s=marker size
# c=vorudiamun ba rangaye rgb
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
