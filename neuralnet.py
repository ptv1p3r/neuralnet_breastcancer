import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras import utils as np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

headers = ['Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
           'concave_points', 'symmetry', 'fractal_dimension']


def importDataset():
    data = pd.read_csv('./dataset/wdbc.data', sep=",", header=None)
    data = data.reset_index(drop=True)
    # data = data.fillna(0)
    # data.describe()

    data_cp = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].copy()
    data_cp.columns = headers
    data_cp = data_cp.reset_index(drop=True)
    # print(data_cp)
    data_diagnosis = data_cp['Diagnosis']
    data_features = data_cp.drop(['Diagnosis'], axis=1)
    # print(data_diagnosis)
    # print(data_features)

    # Splitting DataSet in 80% for training and 20% for testing
    X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_diagnosis, test_size=0.20, random_state=42)

    # re-indexing the subsets
    data_x_train = X_train.reset_index(drop=True)
    data_x_test = X_test.reset_index(drop=True)
    data_y_train = Y_train.reset_index(drop=True)
    data_y_test = Y_test.reset_index(drop=True)

    # converting to numpy arrays to be supported by KERAS API
    data_x_train = data_x_train.values
    data_x_test = data_x_test.values
    data_y_train = data_y_train.values
    data_y_test = data_y_test.values

    rfc = RandomForestClassifier()
    rfc.fit(data_x_train, data_y_train)
    score = rfc.score(data_x_train, data_y_train)
    # print(score)


def createModel():
    model = Sequential()
    model.add(Dense(9, activation='relu', input_dim=9))
    model = Sequential()
    model.add(Dense(9, activation='relu', input_dim=9))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='relu', input_shape=(9,)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', input_shape=(5,)))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    model = createModel()
    model.fit(data_x_train, data_y_train, epochs=500, batch_size=32)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
    scores = model.evaluate([test], Y[test], verbose=0)
    print("{}: {.2f}".format(model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    # # print(data_cp)
    # data_cp_labels = data_cp['Diagnosis']
    # data_cp_features = data_cp.drop(['Diagnosis'], axis=1)
    # print(data_cp_features)
    # data_cp_features = data_cp_features.reset_index(ignore_index=True)


# TODO: Como funciona o SHAPE , e como funciona com texto.


def libs_version():
    print(tf.__version__)
    print(np.__version__)
    print(pd.__version__)
