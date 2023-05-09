from sklearn.cluster import DBSCAN


class DBSCANWrapped:
    def __init__(self, dbscan: DBSCAN):
        self._dbscan = dbscan
        self.labels_ = []

    def get_labels(self):
        print("Calling get")
        return self.labels_

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

    labels_ = property(get_labels, set_labels)

