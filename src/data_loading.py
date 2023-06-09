import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

INLINERS_SIZE = 4750
OUTLIERS_SIZE = 250


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1290)

    return X_train.values, X_test.values, y_train.values, y_test.values


def load_musk(path):
    def map_class(molecule_name):
        cl = molecule_name.split("-")[0]
        if cl == "MUSK":
            return 1
        return 0

    df = pd.read_csv(path, header=None)
    df[0] = df[0].apply(lambda l: map_class(l))
    df = df.drop(columns=[1])

    # downsample classes
    non_musk = df[df[0] == 1]
    musk = df[df[0] != 1]

    non_musk_downsampled = non_musk.sample(n=OUTLIERS_SIZE, random_state=58)
    musk_downsampled = musk.sample(n=INLINERS_SIZE, random_state=12)
    combined = pd.concat([non_musk_downsampled, musk_downsampled])

    X = combined.drop(columns=[0])
    y = combined[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2901)

    return X_train.values, X_test.values, y_train.values, y_test.values
