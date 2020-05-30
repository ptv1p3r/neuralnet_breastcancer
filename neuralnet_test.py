import tensorflow as tf
import pandas as pd
import numpy as np
import os
import os.path
from os import path

# URL da BD
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

# Nome do diretorio do dataset
dataset_path_name = 'dataset'

# Nome do ficheiro da DB
file_name = "wdbc.data"

# Nome que vai ser dado a BD
data_name = 'wdbc.data'


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

# Confirmar os dados
print(my_data)

