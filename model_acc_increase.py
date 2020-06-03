import tensorflow as tf
from config import *
from models import model_sequential
from dataset import dataset

increaseModelAcc = 0
X_train, X_test, Y_train, Y_test = dataset()

if models_path:
    modelGoal = tf.keras.models.load_model(models_path)
    lossGoal, accGoal = modelGoal.evaluate(X_test, Y_test, verbose=2)
    model_sequential(X_train, Y_train)
    modelSequential, history_dict = model_sequential(X_train, Y_train)
    loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
else:
    model_sequential(X_train, Y_train)
    modelSequential, history_dict = model_sequential(X_train, Y_train)
    loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
    quit()

while increaseModelAcc <= 10:
    if acc <= accGoal:
        model_sequential(X_train, Y_train)
        modelSequential, history_dict = model_sequential(X_train, Y_train)
        loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
        increaseModelAcc += 1
    else:
        break

if acc > accGoal:
    modelSequential.save(models_path)
    # print("Test loss: ", loss)
    print("Test accuracy: ", acc)
    print("Model was improved by: ", (acc - accGoal))
else:
    print('Não foi possível melhorar o modelo!')
