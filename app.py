import getopt
import sys
from neuralnet import training, acc_increase, predict


def main(argv):
    try:
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

