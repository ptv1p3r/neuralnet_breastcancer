# import tensorflow as tf
# from config import *
# from models import model_sequential, model_sequential_increase
# from dataset import dataset
# import numpy as np
#
# increaseModelAcc = 1
# X_train, X_test, Y_train, Y_test = dataset()
#
# if modelExists:
#     modelGoal = tf.keras.models.load_model(models_path)
#     lossGoal, accGoal = modelGoal.evaluate(X_test, Y_test, verbose=2)
#     loss = lossGoal
#     acc = accGoal
# else:
#     print('There is no model to improve! Create one by using app.py first.')
#
# while increaseModelAcc <= 200:
#     if acc <= accGoal:
#         for i in np.arange(1, 3, 1):
#             for j in np.arange(15, 60, 5):
#                 for k in np.arange(0.1, 0.5, 0.1):
#                     modelSequential, history_dict = model_sequential_increase(X_train, Y_train, i, j, k)
#                     loss, acc = modelSequential.evaluate(X_test, Y_test, verbose=2)
#                     print('Current Try: ', increaseModelAcc)
#                     increaseModelAcc += 1
#     else:
#         break
#
# if acc > accGoal:
#     modelSequential.save(models_path)
#     print("Test accuracy: ", acc)
#     print("Model was improved by: ", (acc - accGoal))
# else:
#     print('The model could not be improved!')
