import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
headers = ['ID', 'Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']


def importDataset():
    data = pd.read_csv('./dataset/wdbc.data', sep=",", header=None)
    # data = data.reset_index(drop=True)
    # data = data.fillna(0)
    # data.describe()

    data_cp = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].copy()
    data_cp.columns = headers
    print(data_cp)


def libs_version():
    print(tf.__version__)
    print(np.__version__)
    print(pd.__version__)


