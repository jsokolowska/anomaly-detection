import pandas as pd
import numpy as np

MUSK_DIR = "../data/raw/musk"
WINE_PATH = "../data/raw/wine/winequality-red.csv"


def load_wine(path: str) -> (pd.DataFrame, pd.Series):
    """
    Loads and prepares Red Wine Quality Dataset
    :param path: path to read wine dataset csv file
    :return: (X, y)
    """

    def map_to_anomaly(x):
        if x in [3, 4, 8]:
            return 1
        else:
            return 0

    wine = pd.read_csv(path, delimiter=';')
    y = wine['quality'].apply(lambda x: map_to_anomaly(x))
    X = wine.drop("quality", axis=1)

    return X, y


def cblof(X) -> pd.Series:
    """
    Detects anomalies with use of CBLOF algorithm TODO implement
    :param X: DataFrame with "clusters" column with cluster numbers
    :return: Ids of anomalies
    """
    # classify clusters as big or small based on heuristic
    # calculate anomaly score for all the examples
    # return anomalous examples
    pass


class CBLOF:
    """
    Detects anomalies with the use of clustering and CBLOF algorithms

    Parameters
    ----------
    clustering_estimator - sklearn clustering algorithm
    alpha, beta - coefficients for deciding small and large clusters
    """
    def __init__(self, clustering_estimator, alpha, beta):
        self.cluster_labels_ = None
        self.cluster_sizes_ = None
        self.cluster_centers_ = None
        self.clustering_estimator = clustering_estimator
        self.alpha = alpha
        self.beta = beta
        self.anomaly_scores_ = None
        self.n_clusters = None
        self.cluster_centers_ = None

    def fit(self, data):
        self.clustering_estimator.fit(X=data, y=None)
        # Get the labels of the clustering results
        # labels_ is consistent across sklearn clustering algorithms
        self.cluster_labels_ = self.clustering_estimator.labels_
        self.n_clusters = len(np.unique(self.cluster_labels_))

        self.cluster_sizes_ = np.bincount(self.cluster_labels_)

        self._find_centers(data)
        self._categorize_clusters(data.shape[0])
        self.anomaly_scores_ = self._calculate_anomaly_score()

        return self

    def _find_centers(self, data):
        # set center as mean of all samples belonging to it
        n_features = data.shape[1]
        self.cluster_centers_ = np.zeros(self.n_clusters, n_features)
        for cluster in np.unique(self.cluster_labels_):
            members_idx = np.where(self.cluster_labels_ == cluster)
            self.cluster_centers_[cluster, :] = np.mean(data[members_idx])

    def _categorize_clusters(self, n_samples):
        # categorize clusters as big or small based on heuristic
        pass

    def _calculate_anomaly_score(self):
        # calc anomaly scores for all samples
        pass



# todo
# calculate clusters
# name them small or big
# calculate anomaly score
# - CBLOF
# - LDCOF
