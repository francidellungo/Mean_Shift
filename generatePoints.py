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
datasets_dir = 'dataset/3d'


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


def plotSpeedUps(x, y, num_points, save_fig=False):
    # Data for plotting

    fig, ax = plt.subplots()

    plt.xticks(np.arange(len(x)), x)
    # plt.xticks(np.arange(x[0], x[-1], step=(x[-1] - x[0]) / len(x)), x)
    plt.margins(0.02)

    ax.plot(np.arange(len(x)), y, 'o-')

    ax.set(xlabel='Threads', ylabel='Speedup',
           title=' {} points'.format(num_points))
    ax.grid()
    plt.ylim([0, x[-1]])

    if save_fig:
        fig.savefig("docs/figures/speedup_omp_{}.png".format(num_points))
    plt.show()


def plotMultipleSpeedUps(x, y_list, num_points, save_fig=False):
    # Data for plotting
    fig, ax = plt.subplots()

    plt.xticks(np.arange(len(x)), x)
    # plt.xticks(np.arange(x[0], x[-1], step=(x[-1] - x[0]) / len(x)), x)
    plt.margins(0.08)
    for y_idx, y in enumerate(y_list):
        plt.plot([i -1 for i in n_threads], y, label=str(dims[y_idx]))  # , alpha=0.7

    plt.legend(loc=1, bbox_to_anchor=(1,1))

    # ax.plot(np.arange(len(x)), y, 'o-')

    ax.set(xlabel='Threads', ylabel='Speedup',
           title=' {} points'.format(num_points))
    ax.grid()
    plt.ylim([0, x[-1]])
    # plt.xlim([0, x[-1]])

    if save_fig:
        fig.savefig("docs/figures/all_speedups_omp.png")
    plt.show()

""" to generate points"""
# generateDatasets(True)
#
# data, cluster_idx = generatePoints(500, n_features, centers, std)
# plot3dData(data, cluster_idx)
# output_dir = os.path.join(datasets_dir, str(len(data)))
#
# saveData(data, output_dir)

""" Plot old and new points"""

# original dataset
num_points = 100
filename = "dataset/tmp_data/c_{}.csv".format(num_points)

cc = loadData(filename)
plot3dData(cc)
print(len(cc))


# shifted data
shifted_filename = "experiments/ms/cuda/{}.csv".format(num_points)
newp = loadData(shifted_filename)
plot3dData(newp)

# original data with cluster id (--> color)
# new_filename = "dataset/ms/cuda/1000.csv"

new_filename = "experiments/original/{}.csv".format(num_points)
newp = loadData(new_filename)
plot3dData(newp)

# generateDatasets(save_dataset=True)

""" Plot 2d graph tile widths results"""
# # dims = [8, 16, 32, 64, 128, 256, 512, 1024]
# # time_res = [0.026578, 0.011482, 0.008174, 0.008009, 0.007517, 0.008300, 0.008582, 0.015352]

# num_points = 10000
# on_server = True
# source_tw_dir = "experiments/tile_widths/{}_{}.csv".format(num_points, ("s" if on_server else "l"))
# data = loadData(source_tw_dir, skip_header=True)
# dims = [int(d) for d in data[:, 0]]
# time_res = data[:, 4]
# plot2dData(dims, time_res, num_points, False)


""" Plot speedUp graphs"""
num_points = 10000
dims = [100, 500, 1000, 10000]
n_threads = range(1, 9)
filename = "experiments/times/seq_openMP_{}.csv".format(num_points)
# filename = "experiments/times/seq_openMP{}.csv".format(num_points)
data = loadData(filename)
row_len = data[0].shape[0]  # row_len == 8
y_list = []
for idx in range(0, len(dims)):
    time_res = data[:, -1][idx * len(n_threads): idx * len(n_threads) + len(n_threads)]
    # time_res = data[:, -1][-8:]
    speed_ups = [time_res[0] / time for time in time_res]
    # speed_ups.pop(1)
    y_list.append(speed_ups)
    plotSpeedUps(n_threads, speed_ups, dims[idx], save_fig=False)

plotMultipleSpeedUps(n_threads, y_list, num_points, True)


# multiple plot same figure: https://www.w3resource.com/graphics/matplotlib/basic/matplotlib-basic-exercise-5.php
