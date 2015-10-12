#!/usr/bin/env python

import clustering
import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main(filename):
    """ Read Input data """
    data = []
    with open(filename,'rb') as csvfile:
        #skip header
        # _ = csvfile.next()
        for line in csvfile:
            line = line.replace("\n", "")
            temp = line.split('    ')
            key1 = ''
            key2 = ''

            key1 = temp[1]

            if len(temp[2]) > 0:
                key2 = temp[2]
            else:
                key2 = temp[3]
            # data.append([float(key1), float(key2)])
            data.append([key1, key2])

    print data

    num_clusters = 10
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    clusters = km.labels_.tolist()

    print clusters

    #clustering
    # km2 = run_clustering(data, output_dir)


def split_data(data,train_split=0.8):
    data = np.array(data)
    num_train = data.shape[0] * train_split
    npr.shuffle(data)

    return (data[:num_train],data[num_train:])

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='datafile')
    parsed_args = cmdline_parser.parse_args()
    main(parsed_args.filename)
