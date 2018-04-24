#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:33:45 2018

@author: danielreyes
"""

import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')
df = pd.read_csv('Pontos.txt', sep="\s+", header=None)

#http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
cluster = KMeans(n_clusters = 3)
cluster.fit(df)

print(cluster.labels_)

##############################################################################
# Plot the results
for i in set(cluster.labels_):
    index = cluster.labels_ == i
#    print(i)
    plt.plot(df[index,0], df[index,1], 'o')
plt.show()