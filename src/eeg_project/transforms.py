"""Training-fitted signed-log compression of EEG feature magnitudes."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SignedLogFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        scales = []
        for column in X.T:
            values = np.abs(column)
            positive = values[values > 0]
            scales.append(np.median(positive) if positive.size else 1.0)
        self.scale_ = np.asarray(scales)
        return self

    def transform(self, X):
        check_is_fitted(self, "scale_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training")
        return np.sign(X) * (np.logaddexp(np.log(np.maximum(np.abs(X), np.finfo(float).tiny)),
                                         np.log(self.scale_)) - np.log(self.scale_))
