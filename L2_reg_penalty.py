import numpy as np

penalty = 0

for i in np.arange(0, W.shape[0]):
    for j in np.arange(0, W.shape[1]):
        penalty += (W[0][1] ** 2)