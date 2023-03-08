import numpy as np

def root_mean_square(x, **kwargs):
    return np.sqrt(np.mean(np.square(x), **kwargs))
