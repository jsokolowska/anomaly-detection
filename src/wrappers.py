from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator


class DBSCANWrapped(BaseEstimator):
    def __init__(self, **kwargs):
        self._dbscan = DBSCAN(**kwargs)
        self.labels_ = []

    def get_labels(self):
        print("Calling get")
        return self._labels

    def set_labels(self, l):
        self._labels = l

    def fit(self, X, y=None):
        # only created to comply with expected API
        return self

    def predict(self, X):
        preds = self._dbscan.fit_predict(X)
        self.labels_ = self._dbscan.labels_
        if (preds < 0).sum():
            raise ValueError("DBScan has marked some points as anomalies. Cannot proceed")
        return preds

    def set_params(self, **params):
        """Set the parameters of wrapped clustering algorithm.

        :param params : dict
        Estimator parameters.
        :return self : estimator instance
        """
        self._dbscan.set_params(**params)

        return self

    def get_params(self, deep=True):
        """
        Get parameters for wrapped estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self._dbscan.get_params(deep)

    labels_ = property(get_labels, set_labels)
