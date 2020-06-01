from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def models(x_train, y_train):
    # Logistic Regression
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)

    # Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    # Random Forest Classifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    # Model Accuracy sobre os dados de treino
    print('[0] Logistic Regression Training Accuracy:', log.score(x_train, y_train))
    print('[1] Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))
    print('[2] Random Forest Classifier Training Accuracy:', forest.score(x_train, y_train))

    return log, tree, forest
