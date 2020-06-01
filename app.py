from config import *
import tensorflow as tf

from dataset_organization import dataset
from models import model_decision_tree_classifier, model_logistic_regression, model_random_forest_classifier, model_sequential

X_train, X_test, Y_train, Y_test = dataset()

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
results = modelSequential.evaluate(X_test, Y_test, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = modelSequential.predict(X_test[:3])
print('predictions shape:', predictions.shape)
