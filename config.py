import os.path
# ###################### Controlo das msg no terminal relacionadas com o Tensorflow ####################################
# '0' 'DEBUG' [Default] Print all messages
# '1' 'INFO' Filter out INFO messages
# '2' 'WARNING' Filter out INFO & WARNING messages
# '3' 'ERROR' Filter out all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# ################################## Develop Mode ######################################################################
# Mostra no terminal o resultado de cada passo efetuado
debug = False
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

# Nome do diretorio do model
models_path_name = 'models'


# ########################### Opções aplicadas ao nosso modelo e tratamento de dados ###################################
# Defenir se queremos baralhar os dados antes de fazer split para os dados de treino e dados de teste
do_shuffle = False
test_size = 0.25
# ######################################################################################################################

# Define corretamente os caminhos do dataset
app_root = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(app_root, dataset_path_name)
database_path = os.path.join(dataset_path, data_name)

# Define corretamente os caminhos dos modelos
models_path = os.path.join(app_root, models_path_name)
