import pandas as pd


# todo autodownload  & loading for both datasets

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
