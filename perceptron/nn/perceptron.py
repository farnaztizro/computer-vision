import numpy as np

class perceptron:
    # N: The number of columns in our input feature vectors
    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix 
        # store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0

    # X:actual training data
    # y:target output class label
    def fit(self, X, y, epochs=10):
        # apply the bias trick
        X = np.c_[X, np.ones((X.shape[0]))]

        # actual training
        for epoch in np.arange(0, epochs):
            # for each epoch loop over each individual data point
            for (x, target) in zip(X, y):
                # take the dot product
                # pass it through the step func
                p = self.step(np.dot(x, self.W))

                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    # scaling this step by our learning rate alpha
                    self.W += -self.alpha * error * x

    # predict the class labels for a given set of input data(input data must be classified)
    def predict(self, X, addBias=True):
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1’s as the last entry in the feature matrix(bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))
            



