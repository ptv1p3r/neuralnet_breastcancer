# ###################### Controlo das msg no terminal relacionadas com o Tensorflow ####################################
# '0' 'DEBUG' [Default] Print all messages
# '1' 'INFO' Filter out INFO messages
# '2' 'WARNING' Filter out INFO & WARNING messages
# '3' 'ERROR' Filter out all messages
LOGLEVEL = '3'
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
test_size = 0.1
# ######################################################################################################################



