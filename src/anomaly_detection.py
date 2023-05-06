import numpy as np
from pyod.models.cblof import CBLOF
from scipy.spatial.distance import cdist

__all__ = ["ClusterBasedAnomalyDetection"]


class ClusterBasedAnomalyDetection:
    def __init__(self, clustering_estimator, dissimilarity_measure, measure_args: dict = None, contamination=0.9):
        """
        Class for anomaly detection based on clustering

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

        :param contamination: Percent of data to be classified as inliners. 1-threshold will be classified as anomalies.
        """
        self._estimator = clustering_estimator
        self._validate_measure(dissimilarity_measure)
        self._measure = dissimilarity_measure
        # todo actually redo usages
        self._contamination = contamination
        self._computed_threshold = None

        if measure_args is None:
            self._measure_args = {}
        else:
            self._measure_args = measure_args

    def decision_fun(self, X):
        """
        Perform anomaly detection and get dissimilarity scores
        --------------------------------
        :param X: data to perform anomaly detection for
        :return: dissimilarity scores
        """
        if self._measure == "cblof":
            return self._cblof(X)
        if self._measure == "ldcof":
            return self._ldcof(X)

        return self._custom_measure(X)

    def detect(self, X):
        """
        Performs anomaly detection
        --------------------------------
        :param X: data to perform anomaly detection for
        :return: np.ndarray of 0s (not an anomaly) and 1s (an anomaly)
        """
        scores = self.decision_fun(X)
        return self._apply_threshold(scores)

    def predict(self, X):
        #todo implement
        pass

    def _validate_measure(self, dissimilarity_measure):
        if dissimilarity_measure == "cblof":
            return
        elif dissimilarity_measure == "ldcof":
            return
        elif callable(dissimilarity_measure):
            return
        else:
            raise ValueError(f"Expected callable or one of [cblof, ldcof], but got {dissimilarity_measure}")

    def _cblof(self, X):
        # todo handle mismatch in cluster counts & handle detected anomalies - maybe param
        cblof_clf = CBLOF(**self._measure_args, clustering_estimator=self._estimator)
        cblof_clf.fit(X)
        scores = cblof_clf.decision_function(X)
        return scores

    def _ldcof(self, X):
        alpha = 0.9
        if "alpha" in self._measure_args.keys():
            alpha = self._measure_args["alpha"]

        beta = 5
        if "beta" in self._measure_args.keys():
            beta = self._measure_args["beta"]

        return LDCOF(alpha=alpha, beta=beta, clustering_estimator=self._estimator,
                     anomaly_threshold=self._contamination).score(X)

    def _apply_threshold(self, scores):
        self._computed_threshold = np.percentile(scores, 100 * self._contamination)
        return (scores > self._computed_threshold).astype('int').ravel()

    def _custom_measure(self, X):
        self._estimator.fit(X)
        return self._measure(X, self._estimator.labels_)


class LDCOF:
    def __init__(self, alpha, beta, clustering_estimator, anomaly_threshold):
        self.alpha = alpha
        self.beta = beta
        self.clustering_estimator = clustering_estimator
        self.cluster_labels = None
        self.sorted_clusters_idx = None
        self.big_cluster_idx = None
        self.n_clusters = None
        self.n_samples = None
        self.n_features = None
        self.cluster_centers = None
        self.avg_dist = None
        self.scores = None
        self.anomaly_threshold = anomaly_threshold

    def score(self, X):
        self.clustering_estimator.fit(X)

        # helper values
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.cluster_labels = self.clustering_estimator.labels_
        self.n_clusters = len(np.unique(self.cluster_labels))

        self._calc_big_clusters()

        return self._decision_function(X)

    def _calc_big_clusters(self):
        cluster_size = np.bincount(self.cluster_labels)
        # sort from largest
        self.sorted_clusters_idx = np.argsort(cluster_size * -1)

        alpha_cond_idx = []
        beta_cond_idx = []

        for i in range(1, len(np.unique(self.cluster_labels))):
            # sum clusters up to this index
            sizes_sum = np.sum(cluster_size[self.sorted_clusters_idx[:i]])
            if sizes_sum >= self.n_samples * self.alpha:
                alpha_cond_idx.append(i)

            if cluster_size[self.sorted_clusters_idx[i - 1]] / cluster_size[self.sorted_clusters_idx[i]] >= self.beta:
                beta_cond_idx.append(i)

        both = np.intersect1d(alpha_cond_idx, beta_cond_idx)

        if len(both) > 0:
            threshold = both[0]
        elif len(alpha_cond_idx) > 0:
            threshold = alpha_cond_idx[0]
        elif len(beta_cond_idx) > 0:
            threshold = beta_cond_idx[0]
        else:
            raise ValueError("Could not separate into large and small clusters")

        self.big_cluster_idx = self.sorted_clusters_idx[0:threshold]

    def _decision_function(self, X):
        self._calculate_centers(X)
        avg_dist = self._calc_avg_dist_in_cluster(X)
        nearest_c, nearest_idx = self._dist_to_nearest_large(X)
        avg_dist_in_nearest = avg_dist[nearest_idx]
        self.scores = np.divide(avg_dist_in_nearest, nearest_c)
        return self.scores

    def _calculate_centers(self, X):
        if hasattr(self.clustering_estimator, "cluster_centers_"):
            self.cluster_centers = self.clustering_estimator.cluster_centers_
        else:
            # calculate centers as means of all points belonging to the cluster
            self.cluster_centers = np.zeros([self.n_clusters, self.n_features])
            for i in range(0, self.n_clusters):
                self.cluster_centers[i, :] = np.mean(X[np.where(self.cluster_labels == i)], axis=0)

    def _calc_avg_dist_in_cluster(self, X):
        avg_dist = np.zeros([self.n_clusters, ])

        for c in range(0, self.n_clusters):
            points_in_cluster = X[np.where(self.cluster_labels == c)]
            dists = cdist(points_in_cluster, [self.cluster_centers[c]])
            avg_dist[c] = np.average(dists)
        return avg_dist

    def _dist_to_nearest_large(self, X):
        n_large_clusters = len(self.big_cluster_idx)
        dist_all = np.zeros([self.n_samples, n_large_clusters])

        for i in range(0, n_large_clusters):
            cluster_center = self.cluster_centers[self.big_cluster_idx[i]]
            dist_all[:, i] = cdist([cluster_center], X)[0, :]

        return np.min(dist_all, axis=1), self.big_cluster_idx[np.argmin(dist_all, axis=1)]
