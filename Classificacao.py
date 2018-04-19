#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:28:33 2018

@author: danielreyes
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
    
#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')

#Carregando os dados de treino
df_treino = pd.read_csv('train/X_train.txt', sep="\s+", header=None)
label_treino = pd.read_csv('train/Y_train.txt', sep="\s+", header=None)

#Carregando os dados de teste
df_teste = pd.read_csv('test/X_test.txt', sep="\s+", header=None)
label_teste = pd.read_csv('test/Y_test.txt', sep="\s+", header=None)

#Transforma todos os dados em tipo numérico
df_treino = parse_columns(df_treino)
df_teste  = parse_columns(df_teste)


#dividindo o dataset de treino importado em 80% em treino e 20% teste
d_treino, d_teste, r_treino, r_teste = train_test_split(
     df_treino, label_treino, test_size=0.2, random_state=0)

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

dados_balanceados = monta_dataset_balanceado(d_treino, r_treino)

modelo_dt = tree.DecisionTreeClassifier()
modelo_rna = MLPClassifier()

modelo_dt.fit(d_treino, r_treino)
predicoes_dt = modelo_dt.predict(d_teste)

acuracia_dt = accuracy_score(r_teste, predicoes_dt)
print("Acurácia modelo árvore de decisão.: " + str(acuracia_dt))

matriz_confusao_dt = confusion_matrix(r_teste, predicoes_dt)
print(matriz_confusao_dt)

print("\n\n=====================\n\n")

modelo_rna.fit(d_treino, r_treino)
predicoes_rna = modelo_rna.predict(d_teste)

acuracia_rna = accuracy_score(r_teste, predicoes_rna)
print("Acurácia modelo RNA.: " + str(acuracia_rna))

matriz_confusao_rna = confusion_matrix(r_teste, predicoes_rna)
print(matriz_confusao_rna)