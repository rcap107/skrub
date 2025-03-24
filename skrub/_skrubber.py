"""
This transformer should be a stand-alone implementation of the pre-processing
steps performed by the TableVectorizer.

After implementing this, the TableVectorizer should perform pre-processing by
calling this object instead of using its own set of transformers.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class Skrubber(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None):
        del y
        pass

    def transform(self, X):
        pass
