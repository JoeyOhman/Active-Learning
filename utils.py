from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_decision_regions
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np


def train(X_train, y_train):
    clf = LogisticRegression()
    # clf = KNeighborsClassifier()
    # clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(5, 2), max_iter=500)
    # clf = MLPClassifier(hidden_layer_sizes=(10, 20))
    clf.fit(X_train, y_train)
    return clf


def plot_decision_boundary(clf, X, y, title):
    # plot_decision_regions(X, y.astype(int), clf)
    plot_decision_regions(X, y.astype(int), clf)
    plt.title(title)
    plt.show()


def eval_clf(clf, X_val, y_val):
    y_pred = clf.predict(X_val)
    # acc = sklearn.metrics.accuracy_score(y_val, y_pred)

    # y_probs = clf.predict_proba(X_val)
    # log_loss = sklearn.metrics.log_loss(y_val, y_probs, labels=[0, 1, 2])
    f1 = sklearn.metrics.f1_score(y_val, y_pred, average="macro")
    # return acc, log_loss, f1
    return f1
