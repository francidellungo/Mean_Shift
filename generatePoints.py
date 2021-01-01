import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import os
import csv

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# import numpy as np

points = 1000
n_features = 3
centers = 3
std = 1.5
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


def loadData(source_dir, skip_header=False):
    """
    Load dataset from file
    :param source_dir: directory where to load data
    :return: loaded dataset
    """
    # d = np.load(source_dir + '.csv')
    dates = []

    with open(source_dir) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        if skip_header: next(csvReader)
        for row in csvReader:
            dates.append([float(r) for r in row])

    return np.array(dates)
    # return d


def generateDatasets(save_dataset=True, datasets_dir='dataset/3d'):
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
    dimensions = [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000, 1000000]

    # # two dimensional datasets
    # for points in [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]:
    #     generate_dataset(points, n_features=2, centers=3, std=2, file_name=f'2D_data_{points}.csv', output_directory=DATASETS_DIR)

    # three dimensional datasets

    for dim in dimensions:
        data, cluster_id = generatePoints(dim, n_features=n_features, n_clusters=centers, std=std)
        if save_dataset:
            # save dataset points
            output_dir = os.path.join(datasets_dir, str(dim))
            saveData(data, output_dir)
            # save dataset points with generated cluster id
            data_with_cidx = np.hstack((data, np.expand_dims(cluster_id, 1)))
            saveData(data_with_cidx, os.path.join(datasets_dir, 'c_' + str(dim)))


def plot2dData(x, y, num_points, milliseconds=True, save_fig=False):
    # Data for plotting
    # x = [8, 16, 32, 64, 128, 256, 512, 1024]
    fig, ax = plt.subplots()

    plt.xticks(np.arange(len(x)), x)
    # plt.xticks(np.arange(x[0], x[-1], step=(x[-1] - x[0]) / len(x)), x)
    plt.margins(0.02)
    if milliseconds:
        y = [yi * 1000 for yi in y]
        time_unit = "ms"
    else:
        time_unit = "s"

    ax.plot(np.arange(len(x)), y, 'o-')

    ax.set(xlabel='Tile width', ylabel='Gpu time ({})'.format(time_unit),
           title='Variable tile width with {} points'.format(num_points))
    ax.grid()

    if save_fig:
        fig.savefig("docs/figures/var_tw_{}_{}.png".format(num_points, time_unit))
    plt.show()


def plot3dData(data, c='blue'):
    """
    Plot 3d data
    :param data: numpy array of data. dims: [n_points, 3]
    :param c: colour or cluster id (optional)
    :return:
    """

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
# generateDatasets(True)
#
# data, cluster_idx = generatePoints(points, n_features, centers, std)
# plot3dData(data, cluster_idx)
# output_dir = os.path.join(datasets_dir, str(len(data)))
#
# saveData(data, output_dir)

""" Plot old and new points"""

# # original dataset
# filename = "dataset/tmp_data/c_1000000.csv"
#
# cc = loadData(filename)
# plot3dData(cc)
# print(len(cc))
#
#
# # shifted data
# shifted_filename = "experiments/ms/cuda/1000.csv"
# newp = loadData(shifted_filename)
# plot3dData(newp)
#
# # original data with cluster id (--> color)
# # new_filename = "dataset/ms/cuda/1000.csv"
#
# new_filename = "experiments/original/1000.csv"
# newp = loadData(new_filename)
# plot3dData(newp)

# generateDatasets(save_dataset=True)

""" Plot 2d graph tile widths results"""
# dims = [8, 16, 32, 64, 128, 256, 512, 1024]
# time_res = [0.026578, 0.011482, 0.008174, 0.008009, 0.007517, 0.008300, 0.008582, 0.015352]

num_points = 10000
on_server = True
source_tw_dir = "experiments/tile_widths/{}_{}.csv".format(num_points, ("s" if on_server else "l"))
data = loadData(source_tw_dir, skip_header=True)
dims = [int(d) for d in data[:, 0]]
time_res = data[:, 4]
plot2dData(dims, time_res, num_points, False)

# x = [1, 2, 3, 4]
# y = [1, 4, 9, 6]
# labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']
# lab = y
# plt.plot(x, y)
# # You can specify a rotation for the tick labels in degrees or with keywords.
# plt.xticks(x, y)
# # Pad margins so that markers don't get clipped by the axes
# plt.margins(0.2)
# # Tweak spacing to prevent clipping of tick-labels
# plt.subplots_adjust(bottom=0.15)
# plt.show()
