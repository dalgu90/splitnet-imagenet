#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sklearn.cluster as clu
import cPickle as pickle


ckpt_fpath = '../baseline/ResNet50.ckpt'
fc_weight_name = 'fc1000/fc/weights'
imagenet_class_fname = 'synset_words.txt'
output_fname = 'clustering.pkl'

num_cluster1 = 2
num_cluster2 = 5
num_cluster3 = 20

# Spectral clustering
def spectral_clustering(X, n_clusters):
    spectral_clu = clu.SpectralClustering(n_clusters=n_clusters)
    y = spectral_clu.fit_predict(X)
    clusters = [[] for _ in range(n_clusters)]
    for i in range(X.shape[0]):
        clusters[y[i]].append(i)

    cluster_centers = []
    for i in range(n_clusters):
        cluster_centers.append(np.average(X[clusters[i],:], axis=0))
    cluster_centers = np.array(cluster_centers)

    return y, clusters, cluster_centers

# Load ImageNet class names
print('Load ImageNet class names')
with open(imagenet_class_fname) as fd:
    classes = [temp.strip() for temp in fd.readlines()]

# Open TensorFlow ckpt and load the last weight
print('Load tensor: %s' % fc_weight_name)
reader = tf.train.NewCheckpointReader(ckpt_fpath)
weight = reader.get_tensor(fc_weight_name)
weight = weight.transpose()

# Clustering
print('Clustering...\n')
y3, clusters3, centers3 = spectral_clustering(weight, num_cluster3)
y2, clusters2, centers2 = spectral_clustering(centers3, num_cluster2)
y1, clusters1, centers1 = spectral_clustering(centers2, num_cluster1)

output = []
for i1, c1 in enumerate(clusters1):
    print('Cluster %d' % (i1+1))
    temp1 = []
    for i2, ci2 in enumerate(c1):
        print('\tCluster %d-%d' % (i1+1, i2+1))
        c2 = clusters2[ci2]
        temp2 = []
        for i3, ci3 in enumerate(c2):
            print('\t\tCluster %d-%d-%d' % (i1+1, i2+1, i3+1))
            c3 = clusters3[ci3]
            for idx in c3:
                print '\t\t\t%s' % classes[idx]
            temp2.append(c3)
        temp1.append(temp2)
    output.append(temp1)

# Save as pkl file
print('Save as pkl file')
with open(output_fname, 'wb') as fd:
    pickle.dump(output, fd)

print('Done!')
