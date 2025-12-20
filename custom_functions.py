# custom_functions.py
"""
Custom functions used in model pipelines
MUST be imported before loading any pickle files
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def invert_column(X):
    """Custom transformer used in model pipeline"""
    return np.max(X, axis=0) - X + 1

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names_in, selected_features):
        self.feature_names_in = np.array(feature_names_in)
        self.selected_features = selected_features

        # Calculate indices of features to keep
        self.keep_indices_ = [
            i for i, name in enumerate(feature_names_in)
            if name in selected_features
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # FIX: Check if input is a DataFrame and use .iloc
        if hasattr(X, "iloc"):
            return X.iloc[:, self.keep_indices_]

        # Otherwise treat as numpy array/sparse matrix
        return X[:, self.keep_indices_]