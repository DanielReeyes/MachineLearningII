#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:28:33 2018

@author: danielreyes
"""

import pandas as pd
import os
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')

#Parâmetros para utilização do Script
USA_DADOS_BALANCEADOS = False
USA_SELECTION_FEATURE = False
USA_EXPLORACAO_VARIAVEIS = False

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
if USA_EXPLORACAO_VARIAVEIS:
    resumo_dataset(r_treino)

#Se usa dados balanceados (mesma proporção para cada uma das classes)
if USA_DADOS_BALANCEADOS:    
    dados_balanceados = monta_dataset_balanceado(d_treino, r_treino)

    #Como a função de montar o dataset balanceado retorna o dataframe com a coluna rótulo, removemos ela
    d_treino = dados_balanceados.loc[:, dados_balanceados.columns != 'label']
    #sobreescrevemos o dataframe r_treino também com os rótulos
    r_treino = dados_balanceados['label']

if USA_SELECTION_FEATURE:
#   Cria um seletor de variáveis e treina ele para ele aprender as relações entre entrada e rótulo
#   por padrão selecinei 10 variáveis
    seletor = SelectKBest(f_classif, k=10)
    seletor = seletor.fit(d_treino, r_treino)

#   Guarda os índices das 10 variáveis que mais influenciam no rótulo
    idxs_selecionados = seletor.get_support(indices=True)
#   Com os índices podemos pegar diretamente as colunas que nos interessam e guardar no dataframe
    nomes_colunas = df_treino.columns.values[seletor.get_support()]
    d_treino = d_treino[idxs_selecionados]
    d_teste = d_teste[idxs_selecionados]
    
    df_treino = df_treino[idxs_selecionados]
    df_teste = df_teste[idxs_selecionados]

#   Para poder demonstrar em forma de gráfico, vamos pegar o nome das colunas selecionadas e suas
#   respectivas pontuações
#    nomes_colunas = df_treino.columns.values[seletor.get_support()]
    scores = seletor.scores_[seletor.get_support()]
    names_scores = list(zip(nomes_colunas, scores))

#   Transforma em um dataframe para chamar a função de plot.bar()
    df_pontuacao = pd.DataFrame(data = names_scores, columns=['Variavel', 'Score'])
#   Ordena de acordo com a pontuação
    df_pontuacao = df_pontuacao.sort_values(['Score', 'Variavel'], ascending = [False, True])

    df_pontuacao = df_pontuacao.set_index(df_pontuacao['Variavel'])
#   Deleta a coluna com o nome da variável para não apresentar nova barra
    del df_pontuacao['Variavel']
    df_pontuacao.plot.bar()
    

#Declaração dos modelos usados
modelo_dt = tree.DecisionTreeClassifier()
modelo_rna = MLPClassifier()
modelo_nb = GaussianNB()

#Treina o modelo de árvore de decisão
modelo_dt.fit(d_treino, r_treino)

#Guarda os valores preditos pela árvore de decisão
predicoes_dt = modelo_dt.predict(d_teste)
acuracia_dt = accuracy_score(r_teste, predicoes_dt)
print("Acurácia modelo árvore de decisão.: " + str(acuracia_dt))

matriz_confusao_dt = confusion_matrix(r_teste, predicoes_dt)
print("\n=====================\n")
print(matriz_confusao_dt)

print("\n=====================\n")

modelo_rna.fit(d_treino, r_treino.values.ravel())

#Guarda os valores preditos pela rede neural
predicoes_rna = modelo_rna.predict(d_teste)
#Calcula a acurácia do modelo passando os valores preditos e os rótulos verdadeiros
acuracia_rna = accuracy_score(r_teste, predicoes_rna)
print("Acurácia modelo RNA.: " + str(acuracia_rna))

matriz_confusao_rna = confusion_matrix(r_teste, predicoes_rna)
print("\n=====================\n")
print(matriz_confusao_rna)

print("\n=====================\n")

#Guarda os valores preditos pelo Naive Bayes
modelo_nb.fit(d_treino, r_treino)
predicoes_nb = modelo_nb.predict(d_teste)
#Calcula a acurácia do modelo passando os valores preditos e os rótulos verdadeiros
acuracia_nb = accuracy_score(r_teste, predicoes_nb)
print("Acurácia modelo NB.: " + str(acuracia_nb))

matriz_confusao_nb = confusion_matrix(r_teste, predicoes_nb)
print("\n=====================\n")
print(matriz_confusao_nb)

print("\n=====================\n")

#Declara um Ensemble por voto passando por parâmetro os 3 modelos que treinamos antes
ensClassifier = VotingClassifier(estimators=[('rna', modelo_rna), ('dt', modelo_dt), ('nb', modelo_nb)], voting='hard')
ensClassifier = ensClassifier.fit(d_treino, r_treino.values.ravel())

#Guarda os valores preditos pelo Ensemble com 3 modelos (rna, árvore de decisão, naive bayes)
predicoes_ensemble = ensClassifier.predict(d_teste)
#Calcula a acurácia do modelo passando os valores preditos e os rótulos verdadeiros
acuracia_ensemble = accuracy_score(r_teste, predicoes_ensemble)
print("Acurácia ensemble.: " + str(acuracia_ensemble))

matriz_confusao_ensemble = confusion_matrix(r_teste, predicoes_ensemble)
print("\n=====================\n")
print(matriz_confusao_ensemble)

print("\n=====================\n")

cabecalho = ['Label 1', 'Label 2', 'Label 3','Label 4','Label 5','Label 6']

#Até o momento apenas utilizamos os dados de treino
#Agora utilizaremos os dados de treinamento para treinar e teste
#Treina os modelos com os dados de treino
modelo_rna.fit(df_treino, label_treino.values.ravel())
modelo_dt.fit(df_treino, label_treino.values.ravel())
modelo_nb.fit(df_treino, label_treino.values.ravel())
ensClassifier.fit(df_treino, label_treino.values.ravel())

#Agora utilizamos para predizer os dados de testes
teste_predicoes_rna = modelo_rna.predict(df_teste)
teste_predicoes_dt = modelo_dt.predict(df_teste)
teste_predicoes_nb = modelo_nb.predict(df_teste)
teste_predicoes_ensemble = ensClassifier.predict(df_teste)

#Calcula para cada modelo, a acurácia e depois calcula Recall e precision para cada rótulo
teste_acuracia_rna = accuracy_score(label_teste, teste_predicoes_rna)
relatorio(label_teste, teste_predicoes_rna, cabecalho, "RNA", teste_acuracia_rna)

teste_acuracia_dt = accuracy_score(label_teste, teste_predicoes_dt)
relatorio(label_teste, teste_predicoes_dt, cabecalho, "Árvore de Decisão", teste_acuracia_dt)

teste_acuracia_nb = accuracy_score(label_teste, teste_predicoes_nb)
relatorio(label_teste, teste_predicoes_nb, cabecalho, "Naive Bayes",teste_acuracia_nb)

teste_acuracia_ensemble = accuracy_score(label_teste, teste_predicoes_ensemble)
relatorio(label_teste, teste_predicoes_ensemble, cabecalho, "Ensemble", teste_acuracia_ensemble)