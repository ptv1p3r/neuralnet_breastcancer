from config import *

import os.path
from os import path

# ###################### Controlo das msg no terminal relacionadas com o Tensorflow ####################################
# '0' 'DEBUG' [Default] Print all messages
# '1' 'INFO' Filter out INFO messages
# '2' 'WARNING' Filter out INFO & WARNING messages
# '3' 'ERROR' Filter out all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
# ################################## Develop Mode ######################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ver estatisticas
import seaborn as sns


def dataset():
    # verifica se a pasta dataset existe, se não, então é criada
    if not path.exists(dataset_path):
        os.makedirs(dataset_path)

    # se a outra existir ele substitui para prevenir que alguma key existente esteja corrompida
    if not path.exists(database_path) or not path.exists(database_path):
        # Faz o download do data set do URL da BD original e guarda no diretorio com o nome da variavel dataset_path_name
        # isto deixa a app mais dinamica para que no futuro possamos trabalhar outros dados
        file_name = tf.keras.utils.get_file(fname=database_path, origin=data_url)

    # Load dos dados da nossa DB
    my_data = pd.read_csv(database_path, delimiter=',')

    if debug:
        print("")
        print("========== Dados row do dataset ==========")
        print("")
        print(my_data)
        print("")

    my_data.columns = ['id', 'diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness',
                       'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension', 'radius_se',
                       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                       'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worse', 'texture_worst',
                       'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                       'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    if debug:
        # Confirmar as colunas se estão corretas
        print("")
        print("========== Dados row com nomes nas colunas ==========")
        print("")
        print(my_data)
        print("")

    # Os dados podem ter algum tipo de ordenação e isso pode afetar o treino
    # Por exemplo se estiverem ordenados por tamanho e começar no Benigno para o Maligno
    # Podemos estár a dividir para o treino os Benignos e para o teste apenas os Malignos
    # Por isso pode ser do interesse baralhar os dados ou não
    if do_shuffle:
        randomized_data = my_data.reindex(np.random.permutation(my_data.index))
    else:
        randomized_data = my_data

    if debug:
        print("")
        print("========== Shuffle dos dados ==========")
        print("")
        print(randomized_data)
        print("")

    if debug:
        print("")
        print("========== Verificar se existem campos sem informação por coluna ==========")
        print("")
        print(randomized_data.isna().sum())
        print("")

    if plot_graphics:
        # Grafico do n° de cancros Benignos ou Malignos
        # Isto é apresentado quando corrermos o comando plt.show()
        sns.countplot(randomized_data['diagnosis'], label='count')

    # perceber o tipo de campos que temos no dataset
    if debug:
        print("")
        print("========== Tipo de Coluna do dataset ==========")
        print("")
        print(randomized_data.dtypes)
        print("")

    # Index 1 é referente a coluna diagnosis, os ":" é referente a todas as linhas
    # Transformo os valored de M no valor 1 e B no valor 0
    # e volto a guardar a coluna
    labelencoder_Y = LabelEncoder()
    randomized_data.iloc[:, 1] = labelencoder_Y.fit_transform(randomized_data.iloc[:, 1].values)

    if debug:
        print("")
        print("========== Valores de diagnosis passados para binario ==========")
        print("")
        print(randomized_data.iloc[:, 1])
        print("")

    if plot_graphics:
        # Cria um par plot da coluna 1 a 6 sabendo que começa no 0
        # Aplicando o K-means apenas para visualizar e não para ordenar
        # Apenas para termos uma visualização dos clusters
        # Azul é o valor 0(B)
        # Laranja é o valor 1(M)
        sns.pairplot(randomized_data.iloc[:, 1:11], hue='diagnosis')

    if debug:
        # Mostra quais colunas interferem com quais colunas
        print("")
        print("========== Correlação entre colunas ==========")
        print("")
        print(randomized_data.iloc[:, 1:12].corr())
        print("")

    if plot_graphics:
        plt.figure(figsize=(10, 10))
        sns.heatmap(randomized_data.iloc[:, 1:12].corr(), annot=True, fmt='.0%')
        plt.show()

    # Divide os dados em array e em diferentes datasets Y e X
    # Y é a coluna diagnosis
    # X são todas as outras colunas
    # Basicamente a coluna Y diz se o passiente tem Cancro ou não e a coluna X os dados relacionados
    X = randomized_data.iloc[:, 2:32].values
    Y = randomized_data.iloc[:, 1].values

    # Divide os dados em dados de teste e dados de treino usando o train_test_split do sklearn
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

    # Escala os dados para criar uma maior correlação entre eles
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, X_test, Y_train, Y_test
