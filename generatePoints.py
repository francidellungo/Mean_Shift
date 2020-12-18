import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import os
import csv

points = 1000
n_features = 3
centers = 3
std = 2
datasets_dir = 'dataset'


def generatePoints(n_points, n_features, n_clusters, std):
    """
    Generate dataset of n-dimensional points
    :param n_points: total number of points
    :param n_features: i-th point dimension (2 for x,y coordinates; 3 for x,y,z)
    :param n_clusters: number of clusters to generate
    :param std: standard deviation of the clusters
    :return: data: generated data points, y: id of the cluster to which each element of the dataset belongs to
    """
    data, y = datasets.make_blobs(
        n_points, n_features, centers=n_clusters, cluster_std=std, shuffle=True, random_state=1000)
    return data, y


def saveData(data, dest_dir):
    """
    Save dataset to file
    :param data: dataset to be saved
    :param dest_dir: directory where to save data
    :return:
    """
    # save dataset to file
    # np.save(dest_dir, data)
    np.savetxt(dest_dir + ".csv", data, delimiter=",")


def loadData(source_dir):
    """
    Load dataset from file
    :param source_dir: directory where to load data
    :return: loaded dataset
    """
    # d = np.load(source_dir + '.csv')
    dates = []

    with open(source_dir) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            dates.append([float(r) for r in row])

    return np.array(dates)
    # return d


def generateDatasets(save_dataset=True):
    """Datasets with std=1 and well defined clusters"""

    # DATASETS_DIR = '../datasets/different_clusters/'
    #
    # # two dimensional datasets
    # for c in range(1, 6):
    #     generate_dataset(points=10000, n_features=2, centers=c, std=1, file_name=f'2D_data_{c}.csv', output_directory=DATASETS_DIR)
    #
    # # three dimensional datasets
    #
    # for c in range(1, 6):
    #     generate_dataset(points=10000, n_features=3, centers=c, std=1, file_name=f'3D_data_{c}.csv', output_directory=DATASETS_DIR)

    """Datasets with an increasing number of points"""
    dimensions = [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]
    datasets_dir = 'dataset/3d'

    # # two dimensional datasets
    # for points in [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]:
    #     generate_dataset(points, n_features=2, centers=3, std=2, file_name=f'2D_data_{points}.csv', output_directory=DATASETS_DIR)

    # three dimensional datasets

    for dim in dimensions:
        data, cluster_id = generatePoints(dim, n_features=3, n_clusters=centers, std=std)
        if save_dataset:
            output_dir = os.path.join(datasets_dir, str(dim))
            saveData(data, output_dir)


def plot3dData(data, c='blue'):
    """
    Plot 3d data
    :param data: numpy array of data. dims: [n_points, 3]
    :param c: colour or cluster id (optional)
    :return:
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # select x, y, z values from data

    # coord = []
    # for c in range(data.shape[-1]):
    #     coord
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    if data.shape[-1] == 4:
        c = data[:, 3]

    # plot data points
    ax.scatter(xs, ys, zs, c=c)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


""" to generate points"""
# data, cluster_idx = generatePoints(points, n_features, centers, std)
# plot3dData(data, cluster_idx)
# output_dir = os.path.join(datasets_dir, str(len(data)))
#
# saveData(data, output_dir)

""" Plot old and new points"""

# original dataset
filename = "dataset/3d/1000.csv"

cc = loadData(filename)
plot3dData(cc)
print(len(cc))


# shifted data
shifted_filename = "dataset/ms/seq/1000.csv"
newp = loadData(shifted_filename)
plot3dData(newp)

# original data with cluster id (--> color)
# new_filename = "dataset/ms/cuda/1000.csv"
new_filename = "dataset/c_1000.csv"
newp = loadData(new_filename)
plot3dData(newp)

# generateDatasets(save_dataset=True)