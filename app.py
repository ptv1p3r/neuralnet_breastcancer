import sys
from pandas import np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from config import *
import tensorflow as tf
from matplotlib import pyplot as plt
from dataset import dataset
from neuralnet import predict
from models import model_decision_tree_classifier, model_logistic_regression, model_random_forest_classifier, \
    model_sequential, model_sequential_increase
import joblib
import getopt
import sys


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


def training():
    X_train, X_test, Y_train, Y_test = dataset()

    print(X_train[0])

    # Escolher o model para treinar
    # Descomentar o modelo pretendido
    regModel = model_logistic_regression(X_train, Y_train)
    print(
        '###############################################################################################################')
    decModel = model_decision_tree_classifier(X_train, Y_train)
    print(
        '###############################################################################################################')
    classModel = model_random_forest_classifier(X_train, Y_train)
    print(
        '###############################################################################################################')
    # TODO: Save a class model test (ver se é utilizavel ou eliminar depois)

    # database_path = os.path.join(models_path, 'classModel.sav')
    # joblib.dump(classMmodel, database_path)
    print(
        '---------------------------------------------------------------------------------------------------------------')
    modelSequential, history_dict = model_sequential(X_train, Y_train)
    # Avaliação do Modelo Sequencial
    print('')
    loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)

    # # Guardar o modelo feito
    modelSequential.save(models_path) if not modelExists else None

    # Apaga o modelo anterior para testar
    # del modelSequential
    #
    # modelTest = tf.keras.models.load_model(models_path)

    # loss, acc = modelTest.evaluate(X_test, Y_test, verbose=2)
    # print("Test loss: ", loss)
    # print("Test accuracy: ", acc)

    print(
        '---------------------------------------------------------------------------------------------------------------')

    # Teste da avaliação e do predict do modelo sequencial
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = modelSequential.evaluate(X_test, Y_test, batch_size=29)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # Isto é para teste
    print('\n# Generate predictions for 3 samples')
    predictions = modelSequential.predict(X_test[:3])
    print('predictions shape:', predictions.shape)

    # The AUC score is simply the area under the curve which can be calculated with Simpson’s Rule. The bigger the AUC score the better our classifier is.
    # isto é a area a tracejado que aparece no grafico

    # AUC score of testing data
    # TODO : nao se fazem imports a meio do codigo
    # TODO : mais de 10 linhas de codigo, é funcao

    y_test_pred = modelSequential.predict(X_test)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_test_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Testing data AUC: ', auc_keras)

    # ROC curve of testing data

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    # AUC score of training data

    y_train_pred = modelSequential.predict(X_train)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_train, y_train_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Training data AUC: ', auc_keras)

    # ROC curve of training data
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    print('Take a batch of 10 examples from the training data and call model.predict on it.')
    example_batch = X_train[:10]
    example_result = modelSequential.predict(example_batch)
    print(example_result)


def acc_increase():
    increaseModelAcc = 1
    X_train, X_test, Y_train, Y_test = dataset()

    if modelExists:
        modelGoal = tf.keras.models.load_model(models_path)
        lossGoal, accGoal = modelGoal.evaluate(X_test, Y_test, verbose=2)
        loss = lossGoal
        acc = accGoal
    else:
        print('There is no model to improve! Create one by using app.py first.')

    while increaseModelAcc <= 200:
        if acc <= accGoal:
            for i in np.arange(1, 3, 1):
                for j in np.arange(15, 60, 5):
                    for k in np.arange(0.1, 0.5, 0.1):
                        modelSequential, history_dict = model_sequential_increase(X_train, Y_train, i, j, k)
                        loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
                        print('Current Try: ', increaseModelAcc)
                        increaseModelAcc += 1
        else:
            break

    if acc > accGoal:
        modelSequential.save(models_path)
        print("Test accuracy: ", acc)
        print("Model was improved by: ", (acc - accGoal))
    else:
        print('The model could not be improved!')


if __name__ == "__main__":
    main(sys.argv[1:])

