from config import *
import tensorflow as tf

from matplotlib import pyplot as plt

from dataset_organization import dataset
from models import model_decision_tree_classifier, model_logistic_regression, model_random_forest_classifier, \
    model_sequential

X_train, X_test, Y_train, Y_test = dataset()

print(X_train[0])

# Escolher o model para treinar
# Descomentar o modelo pretendido
regModel = model_logistic_regression(X_train, Y_train)
print('###############################################################################################################')
decModel = model_decision_tree_classifier(X_train, Y_train)
print('###############################################################################################################')
classMmodel = model_random_forest_classifier(X_train, Y_train)
print('###############################################################################################################')

print('---------------------------------------------------------------------------------------------------------------')
modelSequential, history_dict = model_sequential(X_train, Y_train)

# Avaliação do Modelo Sequencial
print('')
loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
print("Test loss: ", loss)
print("Test accuracy: ", acc)

# Guardar o modelo feito
modelSequential.save(models_path)

# Apaga o modelo anterior para testar
# del modelSequential
#
# modelTest = tf.keras.models.load_model(models_path)

# loss, acc = modelTest.evaluate(X_test, Y_test, verbose=2)
# print("Test loss: ", loss)
# print("Test accuracy: ", acc)


print('---------------------------------------------------------------------------------------------------------------')

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

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_test_pred = modelSequential.predict(X_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_test_pred)
auc_keras = auc(fpr_keras, tpr_keras)
print('Testing data AUC: ', auc_keras)

# ROC curve of testing data

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# AUC score of training data

y_train_pred = modelSequential.predict(X_train)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_train, y_train_pred)
auc_keras = auc(fpr_keras, tpr_keras)
print('Training data AUC: ', auc_keras)

# ROC curve of training data
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

print('Take a batch of 10 examples from the training data and call model.predict on it.')
example_batch = X_train[:10]
example_result = modelSequential.predict(example_batch)
print(example_result)