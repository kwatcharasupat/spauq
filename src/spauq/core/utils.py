import numpy as np
from numpy.lib.stride_tricks import as_strided


def root_mean_square(x, **kwargs):
    return np.sqrt(np.mean(np.square(x), **kwargs))
