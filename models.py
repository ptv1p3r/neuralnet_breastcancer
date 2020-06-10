import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# Outra maneira de receber as metricas dos modelos
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from config import *
from utils  import structureCheck
import math as math
import numpy as np

APP_ROOT, DATASET_PATH, MODELS_PATH, MODEL_EXISTS, DATASET_FILE = structureCheck()

def model_logistic_regression(x_train, y_train):
    # Logistic Regression
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('Logistic Regression Training Accuracy:', log.score(x_train, y_train))

    getAccuracy(log, x_train, y_train)
    getLogRegression(log, x_train, y_train)
    return log


def model_decision_tree_classifier(x_train, y_train):
    # Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))

    getAccuracy(tree, x_train, y_train)
    getLogRegression(tree, x_train, y_train)

    return tree


def model_random_forest_classifier(x_train, y_train):
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('Random Forest Classifier Training Accuracy:', forest.score(x_train, y_train))

    getAccuracy(forest, x_train, y_train)
    getLogRegression(forest, x_train, y_train)

    return forest


def model_sequential(x_train, y_train):
    # Define a "shallow" logistic regression model
    # Input layer é de 29 neuronios isto corresponde as features do dataset
    # isto conecta a uma unica hiden layer de 15 neuronios escolhidos ao calhas
    # cada hiden layer é ativada pela afunção de ativação  'relu'
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(30,)))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    # isto tudo conecta a uma unica layer de 1 neuronio que tem a função de ativação sigmoid apliacada
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # optimizer = tf.keras.optimizers.Adam(lr=0.010)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Pass several parameters to 'EarlyStopping' function and assign it to 'earlystopper'
    # earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1,
    #                                                 mode='auto')
    # Fit model over 2000 iterations with 'earlystopper' callback, and assign it to history
    history = model.fit(x_train, y_train, epochs=50, validation_split=0.15, verbose=1)

    history_dict = history.history

    return model, history_dict


def model_sequential_increase(x_train, y_train, layers, neurons, dropout):
    print("Numero de Layers : ", layers)
    print("Numero de Neuronios : ", neurons)
    print("Numero de Dropout : ", dropout)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(30,)))
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))

    if layers == 2:
        neurons = math.ceil((neurons / 2))
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
        print("Numero de Layers : ", layers)
        print("Numero de Neuronios : ", neurons)
        print("Numero de Dropout : ", dropout)

    if layers == 3:
        print("Numero de Layers : ", layers)
        print("Numero de Neuronios : ", neurons)
        print("Numero de Dropout : ", dropout)
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
        neurons = math.ceil((neurons/2))
        print("Numero de Layers : ", layers)
        print("Numero de Neuronios : ", neurons)
        print("Numero de Dropout : ", dropout)
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.15, verbose=1)
    history_dict = history.history

    return model, history_dict


def getAccuracy(model, x_train, y_train):
    print('')
    # teste da accuracy do model no data test com a confusion matrix
    # [TP][FP]
    # [FN][TN]
    cm = confusion_matrix(y_train, model.predict(x_train))
    print(cm)
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    print('Testing Accuracy', (TP + TN) / (TP + TN + FN + FP))
    print('')


def getLogRegression(model, x_train, y_train):
    print('')
    print('Model - Logistic Regression')
    print(classification_report(y_train, model.predict(x_train)))
    print(accuracy_score(y_train, model.predict(x_train)))
    print('')


def sequencial_predict(X_test):
    modelSequencial = tf.keras.models.load_model(MODELS_PATH)
    prediction = modelSequencial.predict(X_test)

    return prediction
