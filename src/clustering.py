import numpy as np
from pyod.models.cblof import CBLOF
from scipy.spatial.distance import cdist


class ClusterBasedAnomalyDetection:
    def __init__(self, clustering_estimator, dissimilarity_measure, measure_args: dict = None,
                 handle_outliers: str = "error"):
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
        # TODO  perform clustering and divide clusters into large and small ones based on alpha and beta
        alpha = 0.9
        if "alpha" in self._measure_args.keys():
            alpha = self._measure_args["alpha"]

        beta = 5
        if "beta" in self._measure_args.keys():
            beta = self._measure_args["beta"]

        scores = LDCOF(alpha=alpha, beta=beta, clustering_estimator=self._estimator).decision_function(X)
        pass

    def _custom_measure(self, X):
        self._estimator.fit(X)
        # todo perform clustering
        # apply measure fun & threshold
        scores = self._measure(X, self._estimator.labels_)

        return self._measure(X, self._estimator.labels_)

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
        return self.apply_threshold(scores)

    def apply_threshold(self, scores):
        pass


class LDCOF:
    def __init__(self, alpha, beta, clustering_estimator, anomaly_threshold):
        self.alpha = alpha
        self.beta = beta
        self.clustering_estimator = clustering_estimator
        self.clusters = None
        self.sorted_clusters_idx = None
        self.big_cluster_idx = None
        self.n_clusters = None
        self.n_samples = None
        self.centers = None
        self.avg_dist = None
        self.scores = None
        self.anomaly_threshold = anomaly_threshold

    def score(self, X):
        self.n_samples = X.shape[0]
        self.clustering_estimator.fit(X)
        self.clusters = self.clustering_estimator.labels_
        self.n_clusters = np.unique(self.clusters)

        self._calc_big_clusters()

        # todo
        return []

    def decision_function(self, X):
        self.clustering_estimator.fit(X)

        # helper values
        self.n_samples = X.shape[0]
        self.clusters = self.clustering_estimator.labels_
        self.n_clusters = np.unique(self.clusters)

        self._calc_big_clusters()

        return self._decision_function(X)

    def _calc_big_clusters(self):
        cluster_size = np.bincount(self.clusters)
        # sort from largest
        self.sorted_clusters_idx = np.argsort(cluster_size * -1)

        alpha_cond_idx = []
        beta_cond_idx = []

        for i in range(1, len(np.unique(self.clusters))):
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
        self._calculate_centers()
        avg_dist = self._calc_avg_dist_in_cluster(X)
        nearest_c = self._dist_to_nearest_large(X)

        self.scores = np.divide(avg_dist / nearest_c)

    def _calculate_centers(self):
        #todo
        self.centers = np.zeros([self.n_clusters, ])
        pass

    def _calc_avg_dist_in_cluster(self, X):
        avg_dist = np.zeros([self.n_clusters, ])

        for c in range(1, self.n_clusters):
            points_in_cluster = X[np.where(self.clusters == c)]
            dists = cdist(points_in_cluster, self.centers[c])
            avg_dist[c] = np.average(dists)
        return avg_dist

    def _dist_to_nearest_large(self, X):
        dist_nearest = np.zeros([self.n_clusters, ])
        # for each point todo
        # find distances to all clusters
        # and return minimum
        big_msk = np.isin(self.big_cluster_idx, self.clusters)
        # for big clusters this is the distance to its own cluster

        return dist_nearest
