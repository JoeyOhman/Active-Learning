import numpy as np

# Choose n points to add to training data (data to label)
from data_handler import get_data, split_data
from utils import train, eval_clf, plot_decision_boundary

acquisition_names = ["Passive", "LC", "Margin", "Entropy"]


# Which data point to label?
def acquisition(model, X_unlabeled, acq_idx, batch_size):
    if acq_idx == 0:
        return acquisition_random(batch_size)
    if acq_idx == 1:
        return acquisition_LC(model, X_unlabeled, batch_size)
    elif acq_idx == 2:
        return acquisition_margin(model, X_unlabeled, batch_size)
    elif acq_idx == 3:
        return acquisition_entropy(model, X_unlabeled, batch_size)
    else:
        return acquisition_random(batch_size)


# Passive Learning
def acquisition_random(batch_size):
    return np.arange(batch_size)


def acquisition_LC(model, X_unlabeled, batch_size):
    probs = model.predict_proba(X_unlabeled)
    max_probs = np.max(probs, axis=1)
    # min_max = np.argmin(max_probs)
    indices = np.argsort(max_probs)[0:batch_size]
    return indices
    # return [min_max]


def acquisition_margin(model, X_unlabeled, batch_size):
    probs = model.predict_proba(X_unlabeled)
    diffs = []
    # Could probably be done nicer with an argsort or partition
    for p_vec in probs:
        max_p = 0
        sec_max_p = 0
        for p in p_vec:
            if p > max_p:
                sec_max_p = max_p
                max_p = p
            elif p > sec_max_p:
                sec_max_p = p
        diffs.append(max_p - sec_max_p)

    diffs = np.array(diffs)
    indices = np.argsort(diffs)[0:batch_size]
    return indices
    # return [np.argmin(diffs)]


def acquisition_entropy(model, X_unlabeled, batch_size):
    probs = model.predict_proba(X_unlabeled)
    entropies = - np.sum(probs * np.log(probs + 1e-10), axis=1)
    indices = np.argsort(entropies)[-batch_size:]
    return indices
    # return [np.argmax(entropies)]


'''
def train_eval(X_train, y_train, X_val, y_val):
    clf = train(X_train, y_train)
    return eval_clf(clf, X_train, y_train, X_val, y_val, "Active Learning Decision Boundary")
'''


# Train on n "best" points
def active_learn(X_train, y_train, X_val, y_val, n_start, n_end, batch_size, acq_idx):
    X_train_cpy = np.copy(X_train)
    y_train_cpy = np.copy(y_train)

    init_points = n_start
    X_labeled = X_train_cpy[:init_points]
    y_labeled = y_train_cpy[:init_points]

    X_unlabeled = X_train_cpy[init_points:]
    y_unlabeled = y_train_cpy[init_points:]

    clf = train(X_labeled, y_labeled)
    acc = eval_clf(clf, X_val, y_val)
    accs = [acc]

    # while len(X_labeled) < n:
    for n in range(n_start+batch_size, n_end+1, batch_size):
        # Choose what to label
        indices = acquisition(clf, X_unlabeled, acq_idx, batch_size)
        X_to_add = X_unlabeled[indices]
        y_to_add = y_unlabeled[indices]

        # Add labeled data points
        X_labeled = np.vstack([X_labeled, X_to_add])
        # y_labeled = np.vstack([y_labeled, y_to_add])
        y_labeled = np.concatenate([y_labeled, y_to_add])
        # Remove from unlabeled set
        X_unlabeled = np.delete(X_unlabeled, indices, 0)
        y_unlabeled = np.delete(y_unlabeled, indices, 0)

        # Train from scratch on new labeled dataset
        clf = train(X_labeled, y_labeled)
        acc = eval_clf(clf, X_val, y_val)
        accs.append(acc)

    # print(len(X_labeled))
    # plot_decision_boundary(clf, X_labeled, y_labeled, acquisition_names[acq_idx])
    return accs
    # return eval_clf(clf, X_val, y_val)


def stuff():
    train, val = get_data()
    X_train, y_train, X_val, y_val = split_data(train, val)

    acc = active_learn(X_train, y_train, X_val, y_val, 20, 50, 10, 3)
    print(acc)


if __name__ == '__main__':
    stuff()
