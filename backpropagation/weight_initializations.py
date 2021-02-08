import numpy as np
## Constant Initialization

# C will equal to 0 or 1
W = np.zeros((64, 32))
W = np.ones((64, 32))
W = np.ones((64, 32)) * C

## Uniform and Normal Distributions

# UNIFORM
# range [lower, upper]
# randomly generate 64*32=2048 values
# each value in this range has equal probability
W = np.random.uniform(low=-0.05, high=0.05, size=(64, 32))

# NORMAL
# µ = 0, σ = 0.05
W = np.random.normal(0.0, 0.5, size=(64, 32))

## LeCun Uniform and Normal

# UNIFORM
F_in = 64
F_out = 32
limit = np.sqrt(3 / float(F_in))
W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))

# NORMAL
F_in = 64
F_out = 32
limit = np.sqrt(1 / float(F_in))
W = np.random.normal(0.0, limit, size=(F_in, F_out))

## Glorot/Xavier Uniform and Normal

# NORMAL
# µ = 0
F_in = 64
F_out = 32
limit = np.sqrt(2 / float(F_in + F_out))
W = np.random.normal(0.0, limit, size=(F_in, F_out))

# UNIFORM
F_in = 64
F_out = 32
limit = np.sqrt(6 / float(F_in + F_out))
W = np.random.normal(low=-limit, high=limit, size=(F_in, F_out))

## He et al./Kaiming/MSRA Uniform and Normal

# UNIFORM
F_in = 64
F_out = 32
limit = np.sqrt(6 / float(F_in))
W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))

# NORMAL
# µ = 0 and sigma = radi 2/f_in
F_in = 64
F_out = 32
limit = np.sqrt(2 / float(F_in))
W = np.random.normal(0.0, limit, size=(F_in, F_out))

