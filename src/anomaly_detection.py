import numpy as np
from pyod.models.cblof import CBLOF
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from typing import Union
import logging
import traceback

__all__ = ["ClusterBasedAnomalyDetection"]

from sklearn.base import BaseEstimator


class ClusterBasedAnomalyDetection(BaseEstimator):
    def __init__(self, clustering_estimator: BaseEstimator, dissimilarity_measure: Union[str, callable],
                 alpha: float = 0.9, beta: float = 5, n_clusters: int = 10, contamination: float = 0.1):
        """
        Class for anomaly detection based on clustering

        :param clustering_estimator: Base clustering algorithm to perform data clustering. Should follow standard
        sklearn API conventions - fit() and predict() methods. Should have ``labels_`` attribute. If ``cluster_centers``
        attribute is available it will be used for dissimilarity calculations, otherwise cluster centers will be
        calculated as mean of samples in the cluster.
        Clustering algorithms supporting outlier detection should be considered in a way that ensures no point will be
        marked as anomaly (labels_ attribute cannot have negative numbers), f.e. for DBScan algorithm setting
        min_samples argument to 1.
        #todo decide what to do (ask Natan & check literature)

        :param dissimilarity_measure: One of: "cblof", "cds" or a custom class.
        Custom dissimilarity measure should have fit method taking data samples and cluster labels, and predict and
        decision_function functions that take in only data samples.

        :param measure_args: Additional arguments for cblof or ldcof measures. Ignored if custom measure is used.

        :param contamination: Percent of data to be classified as inliners. 1-threshold will be classified as anomalies.
        """
        self.clustering_estimator = clustering_estimator
        self._validate_measure(dissimilarity_measure)
        self.dissimilarity_measure = dissimilarity_measure
        self.contamination = contamination

        self._chosen_measure = None

        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters

    def set_params(self, **params):
        """Set the parameters of this estimator and nested clustering algorithm.

        :param params : dict
        Estimator parameters.
        :return self : estimator instance
        """
        # set clustering alg params
        clustering_estimator_params = self.clustering_estimator.get_params()
        clustering_params = {k: v for k, v in params.items() if k in clustering_estimator_params.keys()}
        self.clustering_estimator.set_params(**clustering_params)

        current_params = self.get_params()
        new_values = {k: v for k, v in params.items() if k in current_params.keys()}
        super().set_params(**new_values)

        return self

    def fit(self, X, y):
        """
        Performs anomaly detection and sets constants on data
        --------------------------------
        :param X: data to perform anomaly detection for
        :param y: present for compliance with sklearn API
        :return: np.ndarray of 0s (not an anomaly) and 1s (an anomaly)
        """
        try:
            if self.dissimilarity_measure == "cblof":
                self._chosen_measure = CBLOF(alpha=self.alpha, beta=self.beta, n_clusters=self.n_clusters,
                                             clustering_estimator=self.clustering_estimator,
                                             contamination=self.contamination)
            elif self.dissimilarity_measure == "ldcof":
                self._chosen_measure = LDCOF(alpha=self.alpha, beta=self.beta,
                                             clustering_estimator=self.clustering_estimator,
                                             contamination=self.contamination)
            else:
                self._chosen_measure = CustomMeasure(self.clustering_estimator, self.dissimilarity_measure)
            self._chosen_measure.fit(X)
        except Exception as e:
            logging.error(traceback.format_exception(e))
            if type(e) is ValueError:
                raise e
            logging.error(traceback.format_exception(e))
        return self

    def decision_function(self, X):
        """
        Perform anomaly detection and get dissimilarity scores
        --------------------------------
        :param X: data to perform anomaly detection for
        :return: dissimilarity scores
        """
        if not self._chosen_measure:
            raise ValueError("Classificator has not been fitted")
        return self._chosen_measure.decision_function(X)

    def predict(self, X):
        if not self._chosen_measure:
            raise ValueError("Classificator has not been fitted")
        return self._chosen_measure.predict(X)

    def _validate_measure(self, dissimilarity_measure):
        if dissimilarity_measure == "cblof":
            return
        elif dissimilarity_measure == "ldcof":
            return
        elif callable(dissimilarity_measure):
            return
        else:
            raise ValueError(f"Expected callable or one of [cblof, ldcof], but got {dissimilarity_measure}")

    def score(self, X, y):
        y_score = self.decision_function(X)
        return roc_auc_score(y, y_score)


class LDCOF:
    def __init__(self, clustering_estimator, contamination, alpha=0.9, beta=5):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.clustering_estimator = clustering_estimator
        self.contamination = contamination

        # fit results
        self._cluster_labels = None
        self._sorted_clusters_idx = None
        self._big_cluster_idx = None
        self._n_clusters = None
        self._n_samples = None
        self._n_features = None
        self._cluster_centers = None
        self._avg_cluster_dist = None
        self._computed_threshold = None

    def fit(self, X):
        try:
            self.clustering_estimator.fit(X)

            # helper values
            self._n_samples = X.shape[0]
            self._n_features = X.shape[1]
            self._cluster_labels = self.clustering_estimator.labels_
            self._n_clusters = len(np.unique(self._cluster_labels))

            self._set_big_clusters()

            self._calculate_centers(X)
            self._calc_avg_dist_in_cluster(X)
            scores = self._decision_function(X)
            self._calculate_threshold(scores)
        except Exception as e:
            logging.error(traceback.format_exception(e))
            if type(e) is ValueError:
                raise e
            logging.error(traceback.format_exception(e))
        return self

    def decision_function(self, X):
        self._is_fitted()
        return self._decision_function(X)

    def predict(self, X):
        self._is_fitted()
        scores = self._decision_function(X)
        return self._process_scores(scores)

    def _is_fitted(self):
        return self._big_cluster_idx is not None and self._cluster_centers is not None

    def _calculate_threshold(self, scores):
        self._computed_threshold = np.percentile(scores, 100 * (1 - self.contamination))

    def _process_scores(self, scores):
        return (scores > self._computed_threshold).astype('int').ravel()

    def _set_big_clusters(self):
        cluster_size = np.bincount(self._cluster_labels)
        # sort from largest
        self._sorted_clusters_idx = np.argsort(cluster_size * -1)

        alpha_cond_idx = []
        beta_cond_idx = []

        for i in range(1, len(np.unique(self._cluster_labels))):
            # sum clusters up to this index
            sizes_sum = np.sum(cluster_size[self._sorted_clusters_idx[:i]])
            if sizes_sum >= self._n_samples * self.alpha:
                alpha_cond_idx.append(i)

            if cluster_size[self._sorted_clusters_idx[i - 1]] / cluster_size[self._sorted_clusters_idx[i]] >= self.beta:
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

        self._big_cluster_idx = self._sorted_clusters_idx[0:threshold]

    def _decision_function(self, X):
        nearest_c, nearest_idx = self._dist_to_nearest_large(X)
        return np.divide(self._avg_cluster_dist[nearest_idx], nearest_c)

    def _calculate_centers(self, X):
        if hasattr(self.clustering_estimator, "cluster_centers_"):
            self._cluster_centers = self.clustering_estimator.cluster_centers_
        else:
            # calculate centers as means of all points belonging to the cluster
            self._cluster_centers = np.zeros([self._n_clusters, self._n_features])
            for i in range(0, self._n_clusters):
                self._cluster_centers[i, :] = np.mean(X[self._cluster_labels == i], axis=0)

    def _calc_avg_dist_in_cluster(self, X):
        self._avg_cluster_dist = np.zeros([self._n_clusters, ])

        for c in range(0, self._n_clusters):
            points_in_cluster = X[self._cluster_labels == c]
            dists = cdist(points_in_cluster, [self._cluster_centers[c]])
            self._avg_cluster_dist[c] = np.average(dists)

    def _dist_to_nearest_large(self, X):
        if self._big_cluster_idx is None:
            print("AAAA")
        n_large_clusters = len(self._big_cluster_idx)
        n_samples = X.shape[0]
        dist_all = np.zeros([n_samples, n_large_clusters])

        for i in range(0, n_large_clusters):
            cluster_center = self._cluster_centers[self._big_cluster_idx[i]]
            dist_all[:, i] = cdist([cluster_center], X)[0, :]

        return np.min(dist_all, axis=1), self._big_cluster_idx[np.argmin(dist_all, axis=1)]


class CustomMeasure:
    def __init__(self, clustering_estimator, measure):
        # parameters
        self.measure = measure
        self.clustering_estimator = clustering_estimator

        # fit results
        self._is_fitted = False

    def fit(self, X):
        self.clustering_estimator.fit(X)
        self.measure.fit(X, self.clustering_estimator.cluster_labels_)
        self._is_fitted = True
        return self

    def decision_function(self, X):
        if not self._is_fitted:
            raise ValueError("Estimator is not fitted")
        return self.measure.decision_fun(X)

    def predict(self, X):
        return self.measure.predict(X)


if __name__ == "__main__":
    # for testing purposes - to be removed
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.cluster import Birch
    from sklearn.model_selection import StratifiedKFold
    from data_loading import load_wine, load_musk

    WINE_PATH = "./wine/winequality-red.csv"
    MUSK_PATH = "./musk/clean2.data"

    df = pd.DataFrame(index=["musk", "wine"], columns=["birch-lcdof", "birch-cblof", "dbscan-lcdof", "dbscan-cblof"])
    # Params for wine
    param_grid = {
        "alpha": [0.8, 0.9, 0.95],
        "beta": [3, 5, 10],
        "contamination": [0.1, 0.07, 0.05, 0.03],
        "n_clusters": [10, 3, 4, 9, 16, 25],
        "threshold": [0.2, 0.5, 0.7]
    }
    N_SPLITS = 5
    X, y = load_wine(WINE_PATH)
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=91, shuffle=True)
    clustering_algs = [("birch", Birch())]
    measures = ["ldcof", "cblof"]
    datasets = {"musk": load_musk(MUSK_PATH), "wine": load_wine(WINE_PATH)}

    for dataset_name, values in datasets.items():
        X, y = values
        for name, algorithm in clustering_algs:
            for measure in measures:
                cbad = ClusterBasedAnomalyDetection(clustering_estimator=algorithm, dissimilarity_measure=measure)

                search = GridSearchCV(param_grid=param_grid, estimator=cbad, scoring="roc_auc", cv=5, n_jobs=1,
                                      verbose=4)
                search.fit(X, y)
                print(f"[{name}/{measure}/{dataset_name}] Best params: {search.best_params_}")
                print(f"[{name}/{measure}/{dataset_name}] Best score: {search.best_score_}")
                df[f"{name}-{measure}"][dataset_name] = search.best_params_
