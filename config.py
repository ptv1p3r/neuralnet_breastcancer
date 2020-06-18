#  Controlo das msg no terminal relacionadas com o Tensorflow
# '0' 'DEBUG' [Default] Print all messages
# '1' 'INFO' Filter out INFO messages
# '2' 'WARNING' Filter out INFO & WARNING messages
# '3' 'ERROR' Filter out all messages
LOGLEVEL = '3'
DEBUG = False
PLOT_GRAPHICS = False
DATA_URI = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
DATA_NAME = 'wdbc.data'
MODELS_FOLDER = 'models'
DATASET_FOLDER = 'dataset'
DO_SHUFFLE = False
TEST_SIZE = 0.1
TRAINING_EVALUATE_BATCH_SIZE_VALUE = 29
TRAINING_EVALUATE_VERBOSE_VALUE = 2
TRAINING_VERBOSE_VALUE = 2
INCREASE_ACC_ATTEMPTS = 1
INCREASE_ACC_MAX_ATTEMPTS = 200
MIN_LAYERS = 1
MAX_LAYERS = 4
RATE_LAYERS = 1
MIN_NEURONS = 15
MAX_NEURONS = 65
RATE_NEURONS = 5
MIN_DROPOUT = 0.1
MAX_DROPOUT = 0.6
RATE_DROPOUT = 0.1
