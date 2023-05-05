from pyod.models.cblof import CBLOF


class ClusterBasedAnomalyDetection:
    def __init__(self, clustering_estimator, dissimilarity_measure, measure_args: dict = None, handle_outliers: str = "error"):
        """

        :param clustering_estimator: Base clustering algorithm to perform data clustering. Should follow standard
        sklearn API conventions - fit() and predict() methods. Should have ``labels_`` attribute. If ``cluster_centers``
        attribute is available it will be used for dissimilarity calculations, otherwise cluster centers will be
        calculated as mean of samples in the cluster.
        Clustering algorithms supporting outlier detection should be considered in a way that ensures no point will be
        marked as anomaly (labels_ attribute cannot have negative numbers), f.e. for DBScan algorithm setting min_samples
        argument to 1.
        #todo decide what to do (ask Natan & check literature)

        :param dissimilarity_measure: One of: "cblof", "cds" or a function taking in data being clustered and cluster
        information, returing dissimilarity score for each data point.

        :param measure_args: Additional arguments for cblof or ldcof measures. Ignored if callable is passed as dissimilarity
        measure.

        #todo implement this
        :param handle_outliers: How to handle cases in which clustering algorithm returns outliers in cluster detection.
        Possible values: "error" (default) or "remap" (maps each outlier detected by algorithm to its own cluster).
        """
        self._estimator = clustering_estimator
        self._validate_measure(dissimilarity_measure)
        self._measure = dissimilarity_measure
        self._measure_args = measure_args

    def _validate_measure(self, dissimilarity_measure):
        if dissimilarity_measure == "cblof":
            return
        elif dissimilarity_measure == "ldcof":
            return
        elif callable(dissimilarity_measure):
            return
        else:
            raise ValueError(f"Expected callable or one of [cblof, ldcof], but got {type(dissimilarity_measure)}")

    def _cblof(self, X):
        # todo handle mismatch in cluster counts & handle detected anomalies - maybe param
        cblof_clf = CBLOF(**self._measure_args, clustering_estimator=self._estimator)
        cblof_clf.fit(X)
        return cblof_clf.predict(X)

    def _ldcof(self, X):
        pass

    def detect(self, X):
        """
        Performs anomaly detection

        :param X: data to perform anomaly detection for
        :return: np.ndarray of 0s (not an anomaly) and 1s (an anomaly)
        """
        if self._measure == "cblof":
            return self._cblof(X)
        if self._measure == "ldcof":
            return self._ldcof(X)
        # todo custom metric usage
        pass
