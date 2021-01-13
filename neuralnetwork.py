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

