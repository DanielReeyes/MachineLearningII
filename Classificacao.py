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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from functions import *

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')

#Parâmetros para utilização do Script
usa_dados_balanceados = True
usa_selection_feature = True
usa_exploracao_variaveis = True
#variável que guardará o valor minimo aceitável de influência na classe predita (selection feature)
valor_min_influencia = 0.01

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

#Pequena exploração de variáveis, mostra o resumo dos dados de acordo com a classe
if usa_exploracao_variaveis:
    resumo_dataset(r_treino)

#Se usa dados balanceados (mesma proporção para cada uma das classes)
if usa_dados_balanceados:    
    dados_balanceados = monta_dataset_balanceado(d_treino, r_treino)

#Declaração dos modelos usados
modelo_dt = tree.DecisionTreeClassifier()
modelo_rna = MLPClassifier()

modelo_dt.fit(d_treino, r_treino)

if usa_selection_feature:
    feat_sel = modelo_dt.feature_importances_
    #Cria um dataframe com o nome das variáveis e com os valores de importancia para classificação
    importances = pd.DataFrame({'feature':d_treino.columns,'importance':np.round(modelo_dt.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    importances = importances[(importances.T >= valor_min_influencia).any()]
    plota_importancias(importances)
    
    colunas_dt = importances.index
    print("\n\n========== COLUNAS:\n\n")
    print(colunas_dt)
    
    df_sel = monta_dataset_selection(d_treino, colunas_dt)
    
    print("\n\n========== IMPORTANCIAS:\n\n")
    print(importances)
    print("\n\n=====================\n\n")

predicoes_dt = modelo_dt.predict(d_teste)
acuracia_dt = accuracy_score(r_teste, predicoes_dt)
print("Acurácia modelo árvore de decisão.: " + str(acuracia_dt))

matriz_confusao_dt = confusion_matrix(r_teste, predicoes_dt)
print(matriz_confusao_dt)

print("\n\n=====================\n\n")

modelo_rna.fit(d_treino, r_treino)
if usa_selection_feature:    
    print('\n\n DATASET RNA:')
    k_rows = len(importances)
    d_treino = SelectKBest(f_classif, k=k_rows).fit_transform(d_treino, r_treino)
    print(d_treino.shape)


predicoes_rna = modelo_rna.predict(d_teste)
acuracia_rna = accuracy_score(r_teste, predicoes_rna)
print("Acurácia modelo RNA.: " + str(acuracia_rna))

matriz_confusao_rna = confusion_matrix(r_teste, predicoes_rna)
print(matriz_confusao_rna)