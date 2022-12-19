import numpy as np
import torch

# PyTorch stuff
from torch.autograd import Variable

# ------ FUNCTIONS AND CLASSES ------ #


def ma(a, ws=100):
    return [np.mean(a[i:i + ws]) for i in range(0, len(a) - ws)]


def get_RandomBatch(X, batch_size):

    # shuffle joint data
    index = np.random.choice(range(len(X)), size=batch_size, replace=False)
    x_sample = X[index]

    return x_sample


def get_SequentialBatch(X, batch_size, it):

    x_sample = X[batch_size * it: batch_size * (it + 1)]

    return x_sample