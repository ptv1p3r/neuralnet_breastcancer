import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import os
import os.path
from os import path

from sklearn.preprocessing import LabelEncoder

# Ver estatisticas
import seaborn as sns

# ################################## Develop Mode #######################################################################
# Mostra no terminal o resultado de cada passo efetuado
debug = True
plot_graphics = False
# ################################## Directorias #######################################################################
# URL da BD
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

# Nome do diretorio do dataset
dataset_path_name = 'dataset'

# Nome do ficheiro da DB
file_name = "wdbc.data"

# Nome que vai ser dado a BD
data_name = 'wdbc.data'

# ########################### Opções aplicadas ao nosso modelo e tratamento de dados ###################################
# Defenir se queremos baralhar os dados antes de fazer split para os dados de treino e dados de teste
do_shuffle = False

# Define corretamente os caminhos do dataset
app_root = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(app_root, dataset_path_name)
database_path = os.path.join(dataset_path, data_name)

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
    plt.show()


if debug:
    # Mostra quais colunas interferem com quais colunas
    print("")
    print("========== Correlação entre colunas ==========")
    print("")
    print(randomized_data.iloc[:, 1:12].corr())
    print("")
