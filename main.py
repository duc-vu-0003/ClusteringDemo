#!/usr/bin/env python

import clustering
import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import silhouette

def run_clustering(data, output_dir):
    """k-means clustering code"""
    [km2,] = clustering.plot_kmeans(data, filename='%s/faithful_kmeans_k2.png' % output_dir)
    clustering.plot_kmeans(data, ks=(10,), filename='%s/faithful_kmeans_k6.png' % output_dir)
    # train and test data
    # data_train, data_test = split_data(data, train_split=0.8)
    # [kmeans_model_2, kmeans_model_6] = clustering.plot_kmeans(data_train, ks=(2,10), suppress_output=True)
    #
    # print kmeans_model_2
    #
    # print kmeans_model_6
    #
    # clustering.kmeans_predict(data_test, kmeans_model_2,
    #                           filename='{0}/faithful_kmeans_k2_predict.png'.format(output_dir))
    # clustering.kmeans_predict(data_test, kmeans_model_6,
    #                           filename='{0}/faithful_kmeans_k6_predict.png'.format(output_dir))

    return km2

def main(filename, iterations, save_diagnostics, output_dir, burnin):
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
            data.append([float(key1), float(key2)])
            # data.append([key1, key2])
    #skip header
        # _ = csvfile.next()
        # for line in csvfile:
        #     eruption_time, waiting_time = line.split(',')
        #     data.append([float(eruption_time), float(waiting_time)])
    print data
    data = np.array(data)
    print data

    # temp111()

    #clustering
    # km2 = run_clustering(data, output_dir)

    silhouette.silhouetteTest(data)

    # print km2





    #expectation-maximization
    # __run_em(data, output_dir, km2)

    #build bayes fmm model
    # __run_bayesfmm(data, iterations, save_diagnostics, output_dir, burnin, km2)

def temp111():
    x = [1, 5, 1.5, 8, 1, 9]
    y = [2, 8, 1.8, 8, 0.6, 11]

    plt.scatter(x,y)
    plt.show()

def split_data(data,train_split=0.8):
    data = np.array(data)
    num_train = data.shape[0] * train_split
    npr.shuffle(data)

    return (data[:num_train],data[num_train:])

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='datafile')
    cmdline_parser.add_argument('--iterations', action='store',
                                dest='iterations', type=int, default=500)
    cmdline_parser.add_argument('--save_diagnostics', action='store_true', default=False)
    cmdline_parser.add_argument('--output_dir', action='store', default='.')
    cmdline_parser.add_argument('--burnin', action='store', type=int, default=0)

    parsed_args = cmdline_parser.parse_args()

    main(parsed_args.filename, parsed_args.iterations,
         parsed_args.save_diagnostics, parsed_args.output_dir, parsed_args.burnin)
