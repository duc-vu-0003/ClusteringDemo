#!/usr/bin/env python

import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics.cluster.unsupervised import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_recall_fscore_support

data = 'data'
data1 = 'data/cluster1.csv'
data2 = 'data/cluster2.csv'
data3 = 'data/cluster3.csv'
data3result = 'data/cluster3result.csv'

def main():
    oper = -1
    while int(oper) != 0:
        print('**************************************')
        print('Choose one of the following: ')
        print('1 - Exercise 1 - Kmeans')
        print('2 - Exercise 2 - Kmeans')
        print('3 - Exercise 2 - Hierarchical')
        print('4 - Exercise 3 - Kmeans')
        print('5 - Exercise 3 - Hierarchical')
        print('0 - Exit')
        print('**************************************')
        oper = int(input("Enter your options: "))

        if oper == 0:
            exit()
        elif oper == 1:
            test1()
        elif oper == 2:
            test2(True)
        elif oper == 3:
            test2(False)
        elif oper == 4:
            test3(True)
        elif oper == 5:
            test3(False)

def test1():
    n_clusters = 10
    data = []
    with open(data1,'rb') as csvfile:
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
            data.append([key1, key2])

    #convert to numpy array
    data = np.array(data)

    # km = KMeans(10).fit(data)

    clusterer = KMeans(n_clusters)
    cluster_labels = clusterer.fit_predict(data)

    print cluster_labels

    # print km.labels_
    # print km.cluster_centers_
    print silhouette_score(data, cluster_labels, sample_size=5000, metric='euclidean')
    showChartKmeans(clusterer, False, data, n_clusters)

def test2(isKmeans):
    n_clusters = 15
    data = []
    with open(data2,'rb') as csvfile:
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
            data.append([key1, key2])

    #convert to numpy array
    data = np.array(data)

    if isKmeans == True:
        clusterer = KMeans(n_clusters)
        cluster_labels = clusterer.fit_predict(data)

        print cluster_labels
        print silhouette_score(data, cluster_labels, metric='euclidean')
        showChartKmeans(clusterer, False, data, n_clusters)
    else:
        ward = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        ward.fit(data)
        print ward.labels_
        print silhouette_score(data, ward.labels_, metric='euclidean')
        showChartHierarchical(ward, False, data, n_clusters)

def test3(isKmeans):
    n_clusters = 16
    data = []
    with open(data3,'rb') as csvfile:
        for line in csvfile:
            line = line.replace("\n", "")
            temp = line.split('   ')
            temp1 = []
            for item in temp:
                if len(item.strip()) > 0:
                    temp1.append(item.strip())
            data.append(temp1)

    # convert to numpy array
    data = np.array(data)
    print data
    #
    # # km = KMeans(16).fit(data)
    #
    if isKmeans == True:
        clusterer = KMeans(n_clusters)
        cluster_labels = clusterer.fit_predict(data)
        #
        print cluster_labels
        print silhouette_score(data, cluster_labels, metric='euclidean')
        showChartKmeans(clusterer, False, data, n_clusters)

        compareResult(cluster_labels)
    else:
        ward = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        ward.fit(data)

        print ward.labels_
        print silhouette_score(data, ward.labels_, metric='euclidean')
        showChartHierarchical(ward, False, data, n_clusters)

        compareResult(ward.labels_)

def split_data(data,train_split=0.8):
    data = np.array(data)
    num_train = data.shape[0] * train_split
    npr.shuffle(data)
    return (data[:num_train],data[num_train:])

def showChartKmeans(km, need_save, data, k):
    labels = km.labels_
    centroids = km.cluster_centers_
    number_of_figures = 1
    num_rows,num_cols = calc_rows_and_cols(number_of_figures)
    fig, axes = pyplot.subplots(num_rows,num_cols,sharex=True,sharey=True)
    # now plot the clusters
    axis = axes
    plot_kmeans(axis,data,k,labels,centroids)
    axis.set_title('Number of Clusters=%d' % k)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')

    if need_save == True:
        fig.savefig(filename)
    else:
        pyplot.show()

def showChartHierarchical(km, need_save, data, k):
    labels = km.labels_
    number_of_figures = 1
    num_rows,num_cols = calc_rows_and_cols(number_of_figures)
    fig, axes = pyplot.subplots(num_rows,num_cols,sharex=True,sharey=True)
    # now plot the clusters
    axis = axes
    plot_hierarchical(axis ,data,labels, k)
    axis.set_title('Number of Clusters=%d' % k)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')

    if need_save == True:
        fig.savefig(filename)
    else:
        pyplot.show()

def calc_rows_and_cols(number_of_figures):
    num_cols = num_rows = 1
    if number_of_figures > 1:
        num_cols = 2
        num_rows = (number_of_figures/num_cols) + (1 if number_of_figures % num_cols > 0 else 0)
    return num_rows,num_cols

def plot_kmeans(axis,data,k,labels,centroids,alpha=None):
    for i in range(k):
        ds = data[np.where(labels==i)]
        dots = axis.plot(ds[:,0],ds[:,1],'o')
        xs = axis.plot(centroids[i,0],centroids[i,1],'kx')
        pyplot.setp(xs,ms=15.0)
        pyplot.setp(xs,mew=2.0)
        if alpha:
            pyplot.setp(dots,alpha=alpha)
            pyplot.setp(xs,alpha=alpha)

def plot_hierarchical(axis, data, labels, k, alpha=None):
    for i in range(k):
        ds = data[np.where(labels==i)]
        dots = axis.plot(ds[:,0],ds[:,1],'o')
        if alpha:
            pyplot.setp(dots,alpha=alpha)
            pyplot.setp(xs,alpha=alpha)

def compareResult(result):
    result.sort()
    data = []
    with open(data3result,'rb') as csvfile:
        for line in csvfile:
            if len(line.strip()) > 0:
                data.append(int(line) - 1)
    data = np.array(data)
    # print data
    # print result

    print precision_recall_fscore_support(data, result, average='micro')

if __name__ == '__main__':
    main()
