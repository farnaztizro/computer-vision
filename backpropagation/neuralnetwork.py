import numpy as np

class NeuralNetwork:
    # alpha value is applied during the weight update phase
    def __init__(self, layers, alpha):

        # initializes our list of weights for each layer
        self.W = []
        # store layers and alpha
        self.layers = layers
        self.alpha = alpha

        # Our weights list W is empty, so let’s go ahead and initialize it 
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

            # where the input connections need a bias term, but the output does not
            w = np.random.randn(layers[-2] + 1, layers[-1])
            self.W.append(w / np.sqrt(layers[-2]))

    # useful for debugging
    def __repr__(self):
        #  construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    # define sigmoid activation function
    def sigmoid(self, x):

        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):

        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):

        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired # of epochs
        for epoch in np.arange(0, epochs):
            #  For each epoch we’ll loop over each individual data point
            # in our training set, make a prediction on the data point,
            # compute the backpropagation phase, and then update our weight matrix
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):

        # storing the output activations for each layer as our data point x forward propagates through the network
        A = [np.atleast_2d(x)]

        ### forward propagation phase
        # FEEDFORWARD:
        for layer in np.range(0, len(self.W)):

            net = A[layer].dot(self.W[layer])
            # compute the net output applying nonlinear activation function
            out = self.sigmoid(net)
            # add to our list of activations
            # A is the output of the last layer in our network(prediction)
            A.append(out)

        # BACKPROPAGATION

        # the first fase is to compute the difference between our predictions and true target value(error)
        # difference between predicted label A and the ground-truth label y
        error = A[-1] - y
        # apply the chain rule and build a liest od deltas
        # The deltas will be used to update our weight matrices, scaled by the learning rate alpha
        # [-1]: last entry in the list
        D = [error * self.sigmoid_deriv(A[-1])]

        # given the delta for the final layer in the network
        for layer in np.arange(len(A) - 2, 0, -1):

            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since looped over our layers in reverse order we need to reverse deltas
        # tartibe vorudi haye d baraks
        D = D[::-1]
       
        # WEIGHT UPDATE PHASE

        for layer in np.arange(0, len(self.W)):

           # updating weight matrix(actual learning)=gradien descent 
           self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    
    ## make prediction on testing set after network train on a given dataset

    # X:the data points we'll be predicting class labels for
    # addBias:e need to add a column of 1’s to X to perform the bias trick
    def predict(self, X, addBias=True):
        
        # initialize p
        p = np.atleast_2d(X)

        if addBias:
            # insert a column of 1's as the last entry in feature matrix
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            
            '''compute the output prediction by taking dot product
             between the current activation value p and wheight matrix
             then passing the value through the nonlinear activation func'''
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predicted value
        return p

    # calculate loss
    def calculate_loss(self, X, targets):

        # make prediction for the input data point then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss










