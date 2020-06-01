from config import *
from dataset_organization import dataset
from models import models

X_train, X_test, Y_train, Y_test = dataset()

# Correr todos os modelos

model = models(X_train, Y_train)

print(model)

# teste da accuracy do model no data test com a confusion matrix
# [TP][FP]
# [FN][TN]

from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    print('Model', i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    print(cm)
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    print('Testing Accuracy', (TP + TN) / (TP + TN + FN + FP))
    print()

# Outra maneira de receber as metricas dos modelos
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print('Model', i)
    print(classification_report(Y_test, model[i].predict(X_test)))
    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()

# Faz um print da prediction do Random Forest Classifier Model
pred = model[2].predict(X_test)
print(pred)
print('')
print(Y_test)
print('###############################################################################################################')




