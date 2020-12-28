import csv
# import pandas as pd
import math
import os
from generatePoints import plot3dData


def compute_distance(point1, point2):
    return math.dist(point1, point2)


def edit_row(new_dir, color, index):
    csv_file = open(new_dir, 'r')
    read = csv.reader(csv_file)
    csv_file.seek(0)
    r = list(read)
    for k, row in enumerate(r):
        if k == index:
            color = str(color)
            r[k].append(color)
    writer = csv.writer(open(new_dir, 'w'))
    writer.writerows(r)


def is_coloured(new_dir, index, n_features):
    csv_file = open(new_dir, 'r')
    reader = csv.reader(csv_file)
    csv_file.seek(0)
    for k, row in enumerate(reader):
        if k == index:
            if len(row) > n_features:
                return True
            else:
                return False


def color_clusters(shifted_csv, out_csv='experiments/original', threshold=1.5, dataset_dir='dataset/3d'):
    split_dir = shifted_csv.split('/')  # 'dataset/ms/1000.csv'

    with open(os.path.join(dataset_dir, split_dir[-1]), 'r') as orig_csv:  # dataset/3d/1000.csv
        with open(out_csv, 'w') as color_csv:
            with open(shifted_csv, 'r') as shifted_csv:
                or_csv = csv.reader(orig_csv, delimiter=',')
                col_csv = csv.writer(color_csv, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                shift_csv = csv.reader(shifted_csv, delimiter=',')
                for i, row in enumerate(or_csv):
                    if i == 0:
                        n_features = len(row)
                    col_csv.writerow(row)
                color_csv.close()
                color_counter = 0
                orig_csv.seek(0)
                for i, row_shift in enumerate(or_csv):
                    if not is_coloured(out_csv, i, n_features):
                        edit_row(out_csv, color_counter, i)
                        shifted_csv.seek(0)
                        for j, row_shift in enumerate(shift_csv):
                            if j == i:
                                point1 = [float(x) for x in row_shift]
                            if j > i:
                                point2 = [float(x) for x in row_shift]
                                if compute_distance(point1, point2) < threshold:
                                    edit_row(out_csv, color_counter, j)
                    color_counter += 1


if __name__ == '__main__':

    # assign cluster to every original point based on mean shift resulting position
    filename = '1000.csv'
    ms_versions = {0: 'seq', 1: 'openmp', 2: 'cuda'}
    ms_version = ms_versions[2]
    shifted_dir = 'experiments/ms'
    output_dir = 'experiments/original'
    output_csv = 'dataset/c_1000.csv'
    dataset_dir = 'dataset/3d'
    color_clusters(os.path.join(shifted_dir, ms_version, filename), os.path.join(output_dir, filename), threshold=10, dataset_dir=dataset_dir)
    # plot3dData()