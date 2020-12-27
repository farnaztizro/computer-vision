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
        
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learninf rate") 
# 32 data points per mini-batch            
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-baches")
args = vars(ap.parse_args())

# handles generating our 2-class classification problem with 1,000 data points
# adding the bias column, and then performing the training and testing split

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
# matris ro vectorize karde
# satrasho bardar ba y sotun hamaro bkon tu on y sotun
y = y.reshape((y.shape[0], 1))
# insert a column of 1’s
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

#  initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
# initializes a list to keep track of our losses after each epoch
losses = []

# loop over the desired number of epochs
for epoch in np.arrange(0, args["epochs"]):
    # initialize the total loss for the epoch
    epochLoss = []

    # loop over our data in batches
    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):

        preds = sigmoid_activation(batchX.dot(W))
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        # Now that we have the error, we can compute the gradient descent update
        gradient = batchX.T.dot(error)
        # updating our weight matrix based on the gradien
        W += -args["alpha"] * gradient

    # update our loss history
    loss = np.average(epochLoss)
    losses.append(loss)

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] avaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))          


# plot the (testing) classification data

plt.style.use("ggplot")
# chanta chiz ru y nemudar(1bodi bashe nemikhad)
plt.figure()
plt.title("Data")
# s=marker size
# c=vorudiamun ba rangaye rgb
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

