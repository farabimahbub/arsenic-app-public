# -*- coding: utf-8 -*-
"""Shared pipeline pieces, imported by both the trainer and the app so the serialized model reloads cleanly.
Lives in space/ because it must ship with the Hugging Face Space alongside model.joblib."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PearsonCorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of any feature pair with |correlation| > threshold. Same filter used across the study."""
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
