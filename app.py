import os.path
import getopt
import sys
from config import *
from neuralnet import acc_increase, predict, training


def main(argv):
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = LOGLEVEL  # or any {'0', '1', '2'}

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

