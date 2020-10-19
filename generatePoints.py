import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import os
import csv

points = 100
n_features = 3
centers = 3
std = 0.5
datasets_dir = 'dataset'


def generateData(n_points, n_features, n_clusters, std):
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

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            dates.append([float(r) for r in row])

    return np.array(dates)
    # return d


def plotData(data, c='blue'):
    """
    Plot 3d data
    :param data: numpy array of data. dims: [n_points, 3]
    :param c: colour (optional)
    :return:
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # select x, y, z values from data
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    # plot data points
    ax.scatter(xs, ys, zs, c=c)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


data, _ = generateData(points, n_features, centers, std)

output_dir = os.path.join(datasets_dir, str(len(data)))

saveData(data, output_dir)
filename = "dataset/100.csv"

cc = loadData(filename)
print(len(cc))
print(cc[:4])

plotData(cc)

