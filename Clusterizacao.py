# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:36:35 2018

@author: Daniel
"""

import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.metrics import calinski_harabaz_score

RESUMIDO = True

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')
df = pd.read_csv('Pontos.txt', sep="\s+", header=None)
df = df.values

#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

#range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
lst_medias = []

if RESUMIDO:
    melhor_n_cluster_S = -1
    melhor_n_cluster_CH = -1
    
    
    
    for n_clusters in range_n_clusters:
        #Inicializa o agrupador KMeans com a quantidade passada no Loop For e um gerador randomizado    
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        #Treina o agrupador e prediz os grupos com os dados passados
        cluster_labels = clusterer.fit_predict(df)
        #Calcula a média de separação dos grupos formados
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("\nQuantidade de clusters: ", n_clusters, "\nSilhueta média: ", silhouette_avg)
        
        if(melhor_n_cluster_S == -1):
            melhor_n_cluster_S = n_clusters
            melhor_media_S = silhouette_avg
        else:
            if(silhouette_avg > melhor_media_S):
                melhor_n_cluster_S = n_clusters
                melhor_media_S = silhouette_avg
                
        CH = calinski_harabaz_score(df, cluster_labels)
        print("Score Calinski Harabaz.: " + str(CH))

        if(melhor_n_cluster_CH == -1):
            melhor_n_cluster_CH = n_clusters
            melhor_media_CH = CH
        else:
            if(CH > melhor_media_CH):
                melhor_n_cluster_CH = n_clusters
                melhor_media_CH = CH


    print("\nMelhor configuração de clusters para métrica silhouette:", melhor_n_cluster_S)
    range_n_clusters = [melhor_n_cluster_S]
    
    print("\nMelhor configuração de clusters para métrica Calinski Harabasz:", melhor_n_cluster_CH)
    range_n_clusters = [melhor_n_cluster_S]
    
    print("\n")    
        
for n_clusters in range_n_clusters:
#    2 gráficos lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    
    #Inicializa o agrupador KMeans com a quantidade passada no Loop For e um gerador randomizado    
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    #Treina o agrupador e prediz os grupos com os dados passados
    cluster_labels = clusterer.fit_predict(df)
    
    #Calcula a média de separação dos grupos formados
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("Quantidade de clusters: ", n_clusters, "Silhueta média: ", silhouette_avg)
    lst_medias.append(silhouette_avg)
    
    #Computa o valor da silhueta pra cada um dos grupos que o modelo encontrou
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        #Rotula no gráfico cada cluster
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        y_lower = y_upper + 10
    
    ax1.set_title("Gráfico da silhueta para vários clusters.")
    ax1.set_xlabel("Valores da silhueta")
    ax1.set_ylabel("Label do Cluster")
    
    #Linha vertical indicando a média do valor de silhueta para todos os clusters
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  #Limpa o rótulo do eixo y do gráfico 1
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    #gráfico 2 mostrando os dados agrupados
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(df[:, 0], df[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    
    #Rotulando os agrupamentos com o numero deles
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    
    ax2.set_title("Visualização dos dados agrupados.")
    ax2.set_xlabel("Posição X")
    ax2.set_ylabel("Posição Y")
    
    plt.suptitle(("Silhueta gerada pelo KMeans "
                  "com quantidade de clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    
    plt.show()

def plota_grafico(X, range_n_clusters):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.set_size_inches(10, 7)
    
    lst_tmp = X.copy()
    lst_tmp.insert(0,0)
    lst_tmp.insert(0,0)
    
    ax1.set_xlim([2, max(range_n_clusters) + 2])
    ax1.set_ylim([0.3, max(lst_tmp) + 0.02])
    ax1.plot(lst_tmp, c='b', label='Média')
    
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('Silhoutte')
    ax1.legend()
    ax1.grid(linestyle='-')
    
    for i, txt in enumerate(lst_medias):
        ax1.annotate(txt, (range_n_clusters[i], lst_medias[i] - 0.004))

    plt.xticks(range_n_clusters)
            
    plt.title("Progresso de Silhoutte")
    plt.show()

if not RESUMIDO:
    plota_grafico(lst_medias, range_n_clusters)