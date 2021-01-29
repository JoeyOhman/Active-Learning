from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

from data_handler import get_data, split_data
from utils import train, eval_clf


def passive_learn(X_train, y_train, X_val, y_val, n):
    X_train = X_train[:n]
    y_train = y_train[:n]
    clf = train(X_train, y_train)
    return eval_clf(clf, X_train, y_train, X_val, y_val, "Passive Learning Decision Boundary")


def stuff():
    train, val = get_data()
    X_train, y_train, X_val, y_val = split_data(train, val)

    passive_learn(X_train, y_train, X_val, y_val, 10)


if __name__ == '__main__':
    stuff()
