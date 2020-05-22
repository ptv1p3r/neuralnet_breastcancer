import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def importDataset():
    data = pd.read_csv('./dataset/wdbc.data', sep=",", header=None)
    print(data)


def libs_version():
    print(tf.__version__)
    print(np.__version__)
    print(pd.__version__)


