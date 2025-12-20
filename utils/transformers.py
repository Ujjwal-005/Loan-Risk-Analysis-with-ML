# utils/transformers.py
import numpy as np

def invert_column(X):
    return np.max(X, axis=0) - X + 1

