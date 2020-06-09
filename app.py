import os.path
import getopt
import sys
import config
from neuralnet import training, acc_increase, predict


def main(argv):
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.LOGLEVEL  # or any {'0', '1', '2'}

        # Define corretamente os caminhos do dataset
        app_root = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(app_root, config.dataset_path_name)
        database_path = os.path.join(dataset_path, config.data_name)

        # Define corretamente os caminhos dos modelos
        models_path = os.path.join(app_root, config.models_path_name)
        modelExists = os.path.exists(models_path)

        # opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
        opts, args = getopt.getopt(argv, "htip:", ["pstring="])
    except getopt.GetoptError as msg:
        print('error :' + str(msg))
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('app.py -t : Train network -i : Increase Model')
            sys.exit()
        elif opt == '-t':
            training()
            sys.exit()
        elif opt == '-i':
            acc_increase()
            sys.exit()
        elif opt in ("-p", "--pstring"):
            predict(arg)
            sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])

