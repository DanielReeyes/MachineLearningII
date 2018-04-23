#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:07:51 2018

@author: danielreyes
"""
import pandas as pd
import numpy as np
#bibliotecas para gráficos
import matplotlib.pyplot as plt 
from collections import Counter


def parse_columns(x):
    data = x.astype(np.float)
    return data

def monta_dataset_balanceado(x, x_label):
    
    classes = x_label[0]

    #Contando quantas vezes cada classe (nota) apareceu e salvando em uma variável
    counter = Counter(classes)
    WALKING = counter[1]
    WALKING_UPSTAIRS = counter[2]
    WALKING_DOWNSTAIRS = counter[3]
    SITTING = counter[4]
    STANDING = counter[5]
    LAYING = counter[6]
    
    qt_classe_min = min(WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
    
    x['label'] = x_label
    print(x.columns)
    print('WALKING: ' + str(WALKING))
    print('WALKING_UPSTAIRS: ' + str(WALKING_UPSTAIRS))
    print('WALKING_DOWNSTAIRS: ' + str(WALKING_DOWNSTAIRS))
    print('SITTING: ' + str(SITTING))
    print('STANDING: ' + str(STANDING))
    print('LAYING: ' + str(LAYING))
    print("Quantidade de dados da classe minima: " + str(qt_classe_min))
    
    dados_WALKING = x[x['label'] == 1]
    print(dados_WALKING.shape)
    
    dados_WALKING_UPSTAIRS = x[x['label'] == 2]
    print(dados_WALKING_UPSTAIRS.shape)
    
    dados_WALKING_DOWNSTAIRS = x[x['label'] == 3]
    print(dados_WALKING_DOWNSTAIRS.shape)
    
    dados_SITTING = x[x['label'] == 4]
    print(dados_SITTING.shape)
    
    dados_STANDING = x[x['label'] == 5]
    print(dados_STANDING.shape)
    
    dados_LAYING = x[x['label'] == 6]
    print(dados_LAYING.shape)
    
    dados_WALKING = dados_WALKING.head(qt_classe_min)
    dados_WALKING_UPSTAIRS = dados_WALKING_UPSTAIRS.head(qt_classe_min)
    dados_WALKING_DOWNSTAIRS = dados_WALKING_DOWNSTAIRS.head(qt_classe_min)
    dados_SITTING = dados_SITTING.head(qt_classe_min)
    dados_STANDING = dados_STANDING.head(qt_classe_min)
    dados_LAYING = dados_LAYING.head(qt_classe_min)
    
    print("\n\n==============DEPOIS DE BALANCEAR:\n\n")
    print(dados_WALKING.shape)
    print(dados_WALKING_UPSTAIRS.shape)
    print(dados_WALKING_DOWNSTAIRS.shape)
    print(dados_SITTING.shape)
    print(dados_STANDING.shape)
    print(dados_LAYING.shape)
    
    frames = [dados_WALKING, dados_WALKING_UPSTAIRS, dados_WALKING_DOWNSTAIRS, dados_SITTING, dados_STANDING, dados_LAYING]
    df_balanceado = pd.concat(frames)
    
    print("\n\n==============DF DEPOIS DE BALANCEAR:\n\n")
    print(df_balanceado.shape)
    del(x['label'])
    
    return df_balanceado

def resumo_dataset(r_treino):
    print("\n\n===================== Exploração de variáveis: \n\n")
    print(r_treino.columns)
    classes = r_treino[0]

    #Contando quantas vezes cada classe (nota) apareceu e salvando em uma variável
    counter = Counter(classes)
    WALKING = counter[1]
    WALKING_UPSTAIRS = counter[2]
    WALKING_DOWNSTAIRS = counter[3]
    SITTING = counter[4]
    STANDING = counter[5]
    LAYING = counter[6]
    
    # declaração das variáveis para o gráfico de pizza, indicando a quantidade de ocorrencia das classes para os pedaços do gráfico”
    labels = 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING' 
    sizes = [WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING]
    colors = ['green', 'red', 'orange', 'blue', 'yellow', 'pink']
    qtClasses = classes.count()
    tituloGrafico = str("Composição do conjunto de dados de treino: " + str(qtClasses) + " amostras")
    
    
    # Usando o matplot para plotar o gráfico na tela
    fig = plt.figure()
    plt.pie(sizes, labels = labels, colors = colors, shadow = True, startangle = 90, autopct='%1.1f%%')
    plt.title(tituloGrafico)
    plt.show()
    fig.savefig('plot.png', dpi=(200))
    
def plota_importancias(importances):
    #Ordena os valores de maior ao menor, filtra só as variáveis com valores de importância acima de 0.01
    importances = importances.sort_values('importance',ascending=False)
#    importances = importances[(importances.T >= valor_min_influencia).any()]
    importances.plot.bar()

def monta_dataset_selection(dataset, colunas):
    
    df = pd.DataFrame(dataset[colunas], columns = colunas)    
    return df