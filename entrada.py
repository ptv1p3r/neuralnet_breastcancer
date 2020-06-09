from config import *
from dataset import dataset

import tensorflow as tf
from sklearn.model_selection import train_test_split

import argparse
import pandas as pd
from models import model_sequential
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler

# parser = argparse.ArgumentParser(description='Test.')
# parser.add_argument('text', action='store', type=str, help='The text to parse.')
#
# args = parser.parse_args()


def predict(data):
    print('#' * 80)
    # Filtro para filtrar alguns caracteres que possa meter por engano
    # raw_text = eval('"' + args.text.replace('"', '\\"') + '"')
    raw_text = eval('"' + data.replace('"', '\\"') + '"')
    # print(raw_text)
    print('-' * 80)
    # print(args.text)
    print(data)
    print('#' * 80)

    result = [x.strip() for x in raw_text.split(',')]
    X_train, X_test, Y_train, Y_test = dataset()

    # print(result)
    print('*' * 80)
    df = pd.DataFrame(result)

    # df = df.apply(pd.to_numeric)
    df = df.astype(float)

    print(df)
    # print(df.dtypes)

    X = df.values

    sc = StandardScaler()
    X = sc.fit_transform(X)

    dff = pd.DataFrame(X)
    # X.columns = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
    #               'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension', 'radius_se',
    #               'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    #               'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worse', 'texture_worst',
    #               'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    #               'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    print('--' * 80)
    # X = np.array(X)
    # print(X)
    print(dff)
    print('')
    # print(X_train[:1])


    modelSequential = tf.keras.models.load_model(models_path)

    # Avaliação do Modelo Sequencial
    print('')
    loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)

    # AUC score of training data
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    y_train_pred = modelSequential.predict(X_train)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_train, y_train_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Training data AUC: ', auc_keras)

    # TODO: Dividir o valor por 1000
    print('')
    print('teste')
    print('M = 1 e B = 0')
    pred = modelSequential.predict(X_test[:10])
    # pred = modelSequential.predict_on_batch(X)
    print(pred)
    print()
    print(Y_test)

    print(pred.mean())

    print("Benigno" if pred.mean() <= 0.50 else "Maligno")
