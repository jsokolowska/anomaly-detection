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
        self._estimator.fit(X)
        alpha = 0.9
        if "alpha" in self._measure_args.keys():
            alpha = self._measure_args["alpha"]

        beta = 5
        if "beta" in self._measure_args.keys():
            beta = self._measure_args["beta"]

        big_clusters = self._get_big_clusters(alpha, beta, len(X))

        # TODO ref https://www.goldiges.de/publications/Anomaly_Detection_Algorithms_for_RapidMiner.pdf
        # The LDCOF score is defined as the distance to the nearest large cluster
        # as illustrated in Figure 1 divided by the average distance to the cluster center
        # of the elements in that large cluster. The intuition behind this is that when
        # small clusters are considered outlying, the elements inside the small clusters
        # are assigned to the nearest large cluster which becomes its local neighborhood.
        # Thus the anomaly score is computed relative to that neighborhood

        #
        scores = self._decision_function(big_clusters)
        pass

    def _get_big_clusters(self, alpha, beta, n_samples):
        clusters = self._estimator.labels_
        cluster_size = np.bincount(clusters)
        # sort from largest
        sorted_clusters_idx = np.argsort(cluster_size * -1)

        alpha_cond_idx = []
        beta_cond_idx = []

        for i in range(1, len(np.unique(clusters))):
            # sum clusters up to this index
            sizes_sum = np.sum(cluster_size[sorted_clusters_idx[:i]])
            if sizes_sum >= n_samples * alpha:
                alpha_cond_idx.append(i)

            if cluster_size[sorted_clusters_idx[i - 1]] / cluster_size[sorted_clusters_idx[i]] >= beta:
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

        return sorted_clusters_idx[0:threshold]

    def _decision_function(self, X, big_clusters):
        scores = np.zeros([X.shape[0],])
        clusters = self._estimator.labels_

        big_msk = np.isin(big_clusters, clusters)



        return scores

    def _calc_avg_dist_in_cluster(self, big):
        pass

    def _custom_measure(self, X, measure_fun):
        # todo perform clustering
        # apply measure fun & threshold
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


class LCDOF:
    def __init__(self, alpha, beta, clustering_estimator):
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

    def score(self, X):
        self.n_samples = X.shape[0]
        self.clustering_estimator.fit(X)
        self.clusters = self.clustering_estimator.labels_
        self.n_clusters = np.unique(self.clusters)

        self._calc_big_clusters()
        pass

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
        scores = np.zeros([X.shape[0], ])

        big_msk = np.isin(self.big_cluster_idx, self.clusters)

        self._calculate_centers()
        self._calc_avg_dist_in_cluster(X)
        return scores

    def _calculate_centers(self):
        self.centers = np.zeros([self.n_clusters, ])
        pass

    def _calc_avg_dist_in_cluster(self, X):
        self.avg_dist = np.zeros([self.n_clusters, ])

        for c in range(1, self.n_clusters):
            points_in_cluster = X[np.where(self.clusters == c)]
            dists = cdist(points_in_cluster, self.centers[c])
            self.avg_dist[c] = np.average(dists)

    def _dist_to_nearest_large(self, X):
        # for each point
        # find distances to all clusters
        # and return minimum
        big_msk = np.isin(self.big_cluster_idx, self.clusters)
        # for big clusters this is the distance to its own cluster


        pass
