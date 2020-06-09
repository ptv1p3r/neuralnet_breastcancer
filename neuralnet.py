import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from dataset import dataset
from models import model_decision_tree_classifier, model_logistic_regression, model_random_forest_classifier, \
    model_sequential, model_sequential_increase
from utils import structureCheck

APP_ROOT, DATASET_PATH, MODELS_PATH, MODEL_EXISTS, DATASET_FILE = structureCheck()
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = dataset()


def training():
    print(X_TRAIN[0])

    # Escolher o model para treinar
    # Descomentar o modelo pretendido
    regModel = model_logistic_regression(X_TRAIN, Y_TRAIN)
    print(
        '###############################################################################################################')
    decModel = model_decision_tree_classifier(X_TRAIN, Y_TRAIN)
    print(
        '###############################################################################################################')
    classModel = model_random_forest_classifier(X_TRAIN, Y_TRAIN)
    print(
        '###############################################################################################################')
    # TODO: Save a class model test (ver se é utilizavel ou eliminar depois)

    # database_path = os.path.join(models_path, 'classModel.sav')
    # joblib.dump(classMmodel, database_path)
    print(
        '---------------------------------------------------------------------------------------------------------------')
    modelSequential, history_dict = model_sequential(X_TRAIN, Y_TRAIN)
    # Avaliação do Modelo Sequencial
    print('')
    loss, acc = modelSequential.evaluate(X_TEST, Y_TEST, verbose=2)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)

    # # Guardar o modelo feito
    modelSequential.save(MODELS_PATH) if not MODEL_EXISTS else None
    # Apaga o modelo anterior para testar
    # del modelSequential
    #
    # modelTest = tf.keras.models.load_model(models_path)

    # loss, acc = modelTest.evaluate(X_TEST, Y_TEST, verbose=2)
    # print("Test loss: ", loss)
    # print("Test accuracy: ", acc)

    print(
        '---------------------------------------------------------------------------------------------------------------')

    # Teste da avaliação e do predict do modelo sequencial
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = modelSequential.evaluate(X_TEST, Y_TEST, batch_size=29)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # Isto é para teste
    print('\n# Generate predictions for 3 samples')
    predictions = modelSequential.predict(X_TEST[:3])
    print('predictions shape:', predictions.shape)

    # The AUC score is simply the area under the curve which can be calculated with Simpson’s Rule. The bigger the AUC score the better our classifier is.
    # isto é a area a tracejado que aparece no grafico

    # AUC score of testing data
    # TODO : nao se fazem imports a meio do codigo
    # TODO : mais de 10 linhas de codigo, é funcao

    y_test_pred = modelSequential.predict(X_TEST)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TEST, y_test_pred)
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

    y_train_pred = modelSequential.predict(X_TRAIN)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TRAIN, y_train_pred)
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
    example_batch = X_TRAIN[:10]
    example_result = modelSequential.predict(example_batch)
    print(example_result)


def acc_increase():
    increaseModelAcc = 1

    if MODEL_EXISTS:
        modelGoal = tf.keras.models.load_model(MODELS_PATH)
        lossGoal, accGoal = modelGoal.evaluate(X_TEST, Y_TEST, verbose=2)
        loss = lossGoal
        acc = accGoal
    else:
        print('There is no model to improve! Create one by using app.py first.')

    while increaseModelAcc <= 200:
        if acc <= accGoal:
            for layers in np.arange(1, 3, 1):
                for neurons in np.arange(15, 60, 5):
                    for dropout in np.arange(0.1, 0.5, 0.1):
                        modelSequential, history_dict = model_sequential_increase(X_TRAIN, Y_TRAIN, layers, neurons, dropout)
                        loss, acc = modelSequential.evaluate(X_TEST, Y_TEST, verbose=2)
                        print('Current Try: ', increaseModelAcc)
                        increaseModelAcc += 1
        else:
            break

    if acc > accGoal:
        modelSequential.save(MODELS_PATH)
        print("Test accuracy: ", acc)
        print("Model was improved by: ", (acc - accGoal))
    else:
        print('The model could not be improved!')


def predict(data):
    print('#' * 80)
    # Filtro para filtrar alguns caracteres que possa meter por engano
    raw_text = eval('"' + data.replace('"', '\\"') + '"')
    print(raw_text)
    print('-' * 80)
    print(data)
    print('#' * 80)

    result = [x.strip() for x in raw_text.split(',')]

    # print(result)
    print('*' * 80)
    df = pd.DataFrame(result)

    # df = df.apply(pd.to_numeric)
    df = df.astype(float)

    print(df)
    # print(df.dtypes)

    X = df.values

    sc = StandardScaler()
    X = sc.fit_transform(X)

    dff = pd.DataFrame(X)
    # X.columns = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
    #               'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension', 'radius_se',
    #               'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    #               'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worse', 'texture_worst',
    #               'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    #               'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    print('--' * 80)
    # X = np.array(X)
    # print(X)
    print(dff)
    print('')
    # print(X_train[:1])

    modelSequential = tf.keras.models.load_model(MODELS_PATH)

    # Avaliação do Modelo Sequencial
    print('')
    loss, acc = modelSequential.evaluate(X_TEST, Y_TEST, verbose=2)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)

    # AUC score of training data
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    y_train_pred = modelSequential.predict(X_TRAIN)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TRAIN, y_train_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Training data AUC: ', auc_keras)

    # TODO: Dividir o valor por 1000
    print('')
    print('teste')
    print('M = 1 e B = 0')
    pred = modelSequential.predict(X_TEST[:10])
    # pred = modelSequential.predict_on_batch(X)
    print(pred)
    print()
    print(Y_TEST)

    print(pred.mean())

    print("Benigno" if pred.mean() <= 0.50 else "Maligno")
