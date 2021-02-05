import numpy as np

from active_learning import active_learn
from data_handler import get_data, split_data
from plotting import plot_strategies, plot_datasets

NUM_RUNS = 10

n_start = 20
n_end = 360


def compare_strategies(train, val, batch_size, dataset_name):
    passive_res = []
    lc_res = []
    margin_res = []
    entropy_res = []
    badge_res = []

    for i in range(NUM_RUNS):
        np.random.shuffle(train)
        X_train, y_train, X_val, y_val = split_data(train, val)

        # Iterate through AL strategies
        for idx, res_list in enumerate([passive_res, lc_res, margin_res, entropy_res, badge_res]):
            res = active_learn(X_train, y_train, X_val, y_val, n_start, n_end, batch_size, idx)
            res_list.append(res)

    x_range = np.arange(n_start, n_end + 1, batch_size)
    results = [passive_res, lc_res, margin_res, entropy_res, badge_res]
    labels = ["Passive", "LC", "Margin", "Entropy", "Badge"]
    plot_strategies(x_range, results, labels, batch_size, dataset_name)


def compare_datasets(datasets, batch_size):
    balanced_res = []
    skewed_res = []

    for i in range(NUM_RUNS):

        for idx, res_list in enumerate([balanced_res, skewed_res]):
            train, val = datasets[idx]
            np.random.shuffle(train)
            X_train, y_train, X_val, y_val = split_data(train, val)
            # Margin Strategy
            res = active_learn(X_train, y_train, X_val, y_val, n_start, n_end, batch_size, 2)
            res_list.append(res)

    x_range = np.arange(n_start, n_end + 1, batch_size)
    results = [balanced_res, skewed_res]
    labels = ["Balanced", "Skewed"]
    plot_datasets(x_range, results, labels, batch_size, "Margin")


def experiment():
    test_strats = True

    batch_sizes = [1, 10, 60, 120]
    batch_sizes.reverse()
    # Dataset names: Skewed, Balanced

    if test_strats:
        dataset_name = "Skewed"
        train, val = get_data(dataset_name)
        for bs in batch_sizes:
            compare_strategies(train, val, bs, dataset_name)
    else:
        # Test datasets
        # train_balanced, val_balanced = get_data("Balanced")
        # train_skewed, val_skewed = get_data("Skewed")
        datasets = [get_data("Balanced"), get_data("Skewed")]
        for bs in batch_sizes:
            compare_datasets(datasets, bs)


if __name__ == '__main__':
    experiment()
