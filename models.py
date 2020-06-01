import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

# Outra maneira de receber as metricas dos modelos
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#
# def models(x_train, y_train):
#     # Logistic Regression
#     log = LogisticRegression(random_state=0)
#     log.fit(x_train, y_train)
#
#     # Decision Tree
#     tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
#     tree.fit(x_train, y_train)
#
#     # Random Forest Classifier
#     forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
#     forest.fit(x_train, y_train)
#
#     # Model Accuracy sobre os dados de treino
#     print('[0] Logistic Regression Training Accuracy:', log.score(x_train, y_train))
#     print('[1] Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))
#     print('[2] Random Forest Classifier Training Accuracy:', forest.score(x_train, y_train))
#
#     return log, tree, forest

def model_logistic_regression(x_train, y_train):
    # Logistic Regression
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('Logistic Regression Training Accuracy:', log.score(x_train, y_train))

    getAccuracy(log, x_train, y_train)
    getLogRegression(log, x_train, y_train)
    return log


def model_decision_tree_classifier(x_train, y_train):
    # Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))

    getAccuracy(tree, x_train, y_train)
    getLogRegression(tree, x_train, y_train)

    return tree


def model_random_forest_classifier(x_train, y_train):
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('Random Forest Classifier Training Accuracy:', forest.score(x_train, y_train))

    getAccuracy(forest, x_train, y_train)
    getLogRegression(forest, x_train, y_train)

    return forest


def model_sequential(x_train, y_train):
    # TODO: Passar isto para função e organizar melhor o modelo sequencial de maneira a que possamos alterar os parametros mais facilmente
    # Define a "shallow" logistic regression model
    # Input layer é de 29 neuronios isto corresponde as features do dataset
    # isto conecta a uma unica hiden layer de 15 neuronios escolhidos ao calhas
    # cada hiden layer é ativada pela afunção de ativação  'relu'
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(15, input_shape=(29,), activation='relu'))
    # isto tudo conecta a uma unica layer de 1 neuronio que tem a função de ativação sigmoid apliacada
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Pass several parameters to 'EarlyStopping' function and assign it to 'earlystopper'
    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1,
                                                    mode='auto')

    # Fit model over 2000 iterations with 'earlystopper' callback, and assign it to history
    history = model.fit(x_train, y_train, epochs=2000, validation_split=0.15, verbose=0, callbacks=[earlystopper])

    history_dict = history.history

    return model, history_dict


def getAccuracy(model, x_train, y_train):
    print('')
    # teste da accuracy do model no data test com a confusion matrix
    # [TP][FP]
    # [FN][TN]
    cm = confusion_matrix(y_train, model.predict(x_train))
    print(cm)
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    print('Testing Accuracy', (TP + TN) / (TP + TN + FN + FP))
    print('')


def getLogRegression(model, x_train, y_train):
    print('')
    print('Model - Logistic Regression')
    print(classification_report(y_train, model.predict(x_train)))
    print(accuracy_score(y_train, model.predict(x_train)))
    print('')

