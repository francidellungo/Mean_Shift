import csv
# import pandas as pd
import math


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


def color_clusters(shifted_csv, out_csv = None, threshold=1.5):
    split_dir = shifted_csv.split('/')
    if out_csv is None:
        out_csv = split_dir[0] + '/c_' + split_dir[-1]

    # new_dir = split_dir[0] + '/c_' + split_dir[-1]
    with open(split_dir[0] + '/' + split_dir[-1], 'r') as orig_csv:
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
    shift_csv = 'dataset/ms/1000.csv'
    output_csv = 'dataset/c_1000.csv'
    color_clusters(shift_csv, output_csv, threshold=10)
