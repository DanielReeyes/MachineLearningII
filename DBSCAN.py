#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:19:31 2018

@author: danielreyes
"""

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import os
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

# #############################################################################

os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')
X = pd.read_csv('Pontos.txt', sep="\s+", header=None)
X = X.values

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
fig, (ax1) = plt.subplots(1)
fig.set_size_inches(9, 5)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.')

silhouette_avg = silhouette_score(X, labels)
print("Quantidade de clusters: ", n_clusters_, "Silhueta média: ", silhouette_avg)

CH = calinski_harabaz_score(X, labels)

print("Sobre a métrica Calinski Harabaz: \nQuanto maior o valor dessa " +
      "proporção, mais coesos serão os clusters (variação baixa dentro " +
      "do cluster) e mais distintos/separados os clusters individuais " +
      "(variação alta entre clusters).")
      
print("\nScore Calinski Harabaz.: " + str(CH))
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()