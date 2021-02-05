import numpy as np
import matplotlib.pyplot as plt

colors = ["blue", "red", "green", "black"]


def create_cluster(loc, var, num_points, class_index):
    covar_mat = np.identity(len(loc)) * var
    points = np.random.multivariate_normal(loc, covar_mat, num_points)
    class_col = np.full((num_points, 1), class_index)
    points = np.hstack([points, class_col])
    return points


def plot_points(points):
    color_strings = [colors[int(c)] for c in points[:, 2]]
    plt.scatter(points[:, 0], points[:, 1], c=color_strings, alpha=0.3)
    plt.show()


def get_balanced_data(num_points):
    num_classes = 4
    points_per_class = int(num_points / num_classes)

    cl0 = create_cluster((1.5, 3), 1, points_per_class, 0)
    cl1 = create_cluster((-0, -1), 1, points_per_class, 1)
    cl2 = create_cluster((-2, -4), 1, points_per_class, 2)
    # cl3 = create_cluster((5, -3), 1, int(points_per_class / 2), 2)
    # cl4 = create_cluster((-5, -1), 1, points_per_class, 3)
    cl4 = create_cluster((-5, -1), 1, points_per_class, 3)
    data = np.vstack([cl0, cl1, cl2, cl4])

    return data


def get_skewed_data(num_points):

    cl0 = create_cluster((1.5, 3), 1, int(num_points*0.5), 0)
    cl1 = create_cluster((-0, -1), 1, int(num_points*0.2), 1)
    cl2 = create_cluster((-2, -4), 1, int(num_points*0.2), 2)
    # cl3 = create_cluster((5, -3), 1, int(points_per_class / 2), 2)
    # cl4 = create_cluster((-5, -1), 1, points_per_class, 3)
    cl4 = create_cluster((-5, -1), 1, int(num_points*0.1), 3)
    data = np.vstack([cl0, cl1, cl2, cl4])

    return data


def get_skewed_difficult_data(num_points):
    cl0 = create_cluster((1.5, 3), 2, int(num_points * 0.5), 0)
    cl1 = create_cluster((-0, -1), 1, int(num_points * 0.2), 1)
    cl2 = create_cluster((-2, -4), 2, int(num_points * 0.2), 2)
    # cl3 = create_cluster((5, -3), 1, int(points_per_class / 2), 2)
    cl3 = create_cluster((-5, -1), 1, int(num_points * 0.05), 3)
    cl4 = create_cluster((6, 9), 1, int(num_points * 0.05), 3)
    data = np.vstack([cl0, cl1, cl2, cl3, cl4])

    return data


def get_data(dataset_name):
    # num_points = 1200
    num_points = 2000

    if dataset_name == "Balanced":
        data = get_balanced_data(num_points)
    elif dataset_name == "Skewed":
        data = get_skewed_difficult_data(num_points)
        # data = get_skewed_data(num_points)
    else:
        print("Dataset name should be Balanced or Skewed")
        return None

    np.random.shuffle(data)
    train_size = int(num_points * 0.7)
    train = data[:train_size]
    val = data[train_size:]

    return train, val


def split_data(train, val):
    X_train = train[:, 0:-1]
    y_train = train[:, -1]

    X_val = val[:, 0:-1]
    y_val = val[:, -1]

    return X_train, y_train, X_val, y_val


def get_random_batch(data, n):
    return data[:n]


def plot_data():
    train, val = get_data("Skewed")
    # X_train, y_train, X_val, y_val = get_data()
    plot_points(train)


if __name__ == '__main__':
    plot_data()
