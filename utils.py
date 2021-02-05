from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_decision_regions
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

from ann import FFNet
from data_handler import get_data, split_data, plot_points


def train(X_train, y_train):
    num_features = X_train.shape[1]
    # num_classes = int(max(y_train) + 1)
    num_classes = 4
    clf = FFNet(num_features, num_classes)
    # clf = LogisticRegression()
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


# In BADGE, X is grads
def k_means_pp_seed(X, k):
    # init_point = X[np.random.randint(len(X))]
    init_point = np.random.randint(len(X))
    clusters = [init_point]
    min_dists = np.linalg.norm(X - clusters[0], axis=1, keepdims=True)
    for t in range(1, k):
        dists = np.linalg.norm(X - X[clusters[-1]], axis=1, keepdims=True)
        dists = np.square(dists)
        dists = np.hstack([min_dists, dists])
        min_dists = np.min(dists, axis=1, keepdims=True)
        probs = min_dists / np.sum(min_dists)
        next_cluster = np.random.choice(np.arange(len(X)), p=np.squeeze(probs))
        clusters.append(next_cluster)

    return np.array(clusters)


if __name__ == '__main__':
    train, val = get_data("Balanced")
    X_train, y_train, X_val, y_val = split_data(train, val)
    # plot_points(train)
    clusts = k_means_pp_seed(X_train, 4)
    cluster_points = X_train[clusts]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c="purple", s=1000)
    rand_clusts = np.random.randint(len(X_train), size=4)
    rand_cluster_points = X_train[rand_clusts]
    plt.scatter(rand_cluster_points[:, 0], rand_cluster_points[:, 1], c="yellow", s=1000)
    plot_points(train)
