import matplotlib.pyplot as plt
import numpy as np


strategy_colors = {
    "Passive": "blue",
    "LC": "orange",
    "Margin": "green",
    "Entropy": "red"
}

dataset_markers = {
    "Balanced": "o",
    "Skewed": "x"
}


def mean_std(rows):
    return np.mean(rows, axis=0), np.std(rows, axis=0)


def aggregate_data(data):
    means, stds = [], []
    for d in data:
        m, s = mean_std(d)
        means.append(m)
        stds.append(s)
    return means, stds


def plot_curves(x, res, batch_size, labels, colors, markers):
    assert len(res) == len(labels)
    means, stds = aggregate_data(res)
    num_points = len(means)

    for i in range(num_points):
        lbl = labels[i]
        plt.plot(x, means[i], label=lbl, marker=markers[i], markersize=4, color=colors[i])
        plt.fill_between(x, means[i] - stds[i], means[i] + stds[i], alpha=0.2, color=colors[i])

    plt.xlabel("Labeled Samples")
    # plt.ylabel("Validation Accuracy")
    plt.ylabel("Test F1")
    plt.title("Learning Curves, $n_{batch} = " + str(batch_size) + "$")
    plt.rcParams["legend.loc"] = 'lower right'
    plt.legend()
    plt.show()


# res contains results of same model and dataset, with different AL strategies
def plot_strategies(x, res, legend_labels, batch_size, dataset_name):

    colors = [strategy_colors[strat] for strat in legend_labels]
    markers = [dataset_markers[dataset_name]] * len(legend_labels)

    plot_curves(x, res, batch_size, legend_labels, colors, markers)


# res contains results of same AL strategy and model, with different datasets
def plot_datasets(x, res, legend_labels, batch_size, strategy_name):

    colors = [strategy_colors[strategy_name]] * len(legend_labels)
    markers = [dataset_markers[dataset_name] for dataset_name in legend_labels]

    plot_curves(x, res, batch_size, legend_labels, colors, markers)


if __name__ == '__main__':
    pass
