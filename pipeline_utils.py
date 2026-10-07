# -*- coding: utf-8 -*-
"""Transformers the fitted models need at load time.

joblib resolves a pickled class by its module path, so the app imports this from the same place the
trainer did. It ships beside the model files.
"""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PearsonCorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops one feature of any pair whose absolute correlation exceeds the threshold."""
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        cm = Xd.corr().abs()
        up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
        self.drop_ = [c for c in up.columns if any(up[c] > self.threshold)]
        self.keep_ = [i for i, c in enumerate(Xd.columns) if c not in self.drop_]
        return self

    def transform(self, X):
        return pd.DataFrame(X).iloc[:, self.keep_].to_numpy()
