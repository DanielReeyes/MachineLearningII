#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 08:48:58 2018

@author: danielreyes
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from pandas import concat
from pandas import DataFrame

#"Constantes" - Para organizar melhor a parametrização do script
RANGE_N = 11.0
TAM_PASSO = 0.01
DIVISOR = 100
QT_PASSO_PRED = 10
TAM_JANELA = 15

#Função para criar um data set e posteriormente dividi-lo em treino e teste
#Dentro da função há a equação que produzirá os valores
def monta_dataset(RANGE_N, TAM_PASSO):
    pontos = []
    valores_n = []
    n_passos = int(RANGE_N / TAM_PASSO)
    for n in range(n_passos):
        result = math.sqrt(1 + (math.sin(n + (math.sin(n) ** 2))))
        pontos.append(result)
        value_n = n/DIVISOR
        valores_n.append(value_n)
    return valores_n, pontos

def monta_cabecalho_janela():
    janela = []
    
    for i in np.arange(TAM_JANELA, -1, -1):
        if(i==0):
            janela = np.append(janela, "x")
        else:
            janela = np.append(janela, "x-"+str(i))

    return janela

def plota_grafico_2(X, y_true, y_predict, titulo, passos_preditos):
    
    y_lim_inf = (RANGE_N / TAM_PASSO) - QT_PASSO_PRED
    y_lim_sup = (RANGE_N / TAM_PASSO)
#    print("Limite Inferior: " + str(y_lim_inf))
#    print("Limite Superior: " + str(y_lim_sup))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, c='b', label='Real')
    ax1.plot(np.arange(y_lim_inf, y_lim_sup), y_predict, c='r', label='Predito', linewidth=1.5)
    ax1.set_xlabel('Passo')
    ax1.set_ylabel('Valor')
    ax1.legend()
    
    plt.title(titulo)
    plt.show()

def metricas(y_real, predicoes):
    #Calcula a eficiência do modelo
    print("Mean squared error: %.2f" % mean_squared_error(y_real, predicoes))
    
    #Calcula a métrica de variância de predições
    print('Variance score: %.2f' % r2_score(y_real, predicoes))

X, y = monta_dataset(11, 0.01)

df_pontos = DataFrame(pd.Series(y).values)

df_pontos = concat([df_pontos.shift(1), 
                      df_pontos.shift(2), 
                      df_pontos.shift(3),  
                      df_pontos.shift(4),
                      df_pontos.shift(5),
                      df_pontos.shift(6),
                      df_pontos.shift(7),
                      df_pontos.shift(8),
                      df_pontos.shift(9),
                      df_pontos.shift(10),
                      df_pontos.shift(11),
                      df_pontos.shift(12),
                      df_pontos.shift(13),
                      df_pontos.shift(14),
                      df_pontos.shift(15),
                      df_pontos], axis=1)

#Monta o cabeçalho contendo os nomes de colunas e seta no dataframe
df_pontos.columns = monta_cabecalho_janela()
#Depois de fazer o shift, removemos as primeiras linhas pois contém valores NaN
df_pontos.drop(df_pontos.head(TAM_JANELA).index,inplace=True)

#Cria o modelo e seta os parâmetros dele
rna = MLPRegressor(hidden_layer_sizes=(100, 100), 
                   activation='tanh', 
                   early_stopping=True)

#print(df_pontos.head(2))
#Cria um dataframe apenas com as variáveis, excluindo os rótulos do momento 'x' a ser predito posteriormente
features = df_pontos.drop('x', axis=1)
#print(df_pontos_x.head(2))
#Cria um dataframe\lista com os rótulos reais
labels = df_pontos['x']

#treina o modelo com o dataframe de variáveis e os rótulos
rna.fit(features, labels)

#Simula um dataset de teste, porém com os dados de treinos, pegando os últimos X dados que seriam testados
features_to_test_model = features.tail(QT_PASSO_PRED)
predicoes = rna.predict(features_to_test_model)

plota_grafico_2(labels.tail(QT_PASSO_PRED),
              labels.tail(QT_PASSO_PRED), 
              predicoes, 
              "Predição de " + str(QT_PASSO_PRED) + " passo(s) \n com deslocamento temporal", 
              QT_PASSO_PRED)

#Agora mensura o quanto o modelo acertou com os dados de treino
metricas(labels.tail(QT_PASSO_PRED), predicoes)

#Criando uma lista pra guardar os valores preditos para cada passo
predicoes_passos = []
#Criando um dataset para guardar os valores que foram utilizados para predizer os próximos N passos
df_teste = DataFrame()

df_treino = features.head(len(X) - QT_PASSO_PRED-TAM_JANELA)
y_treino = labels.head(len(X) - QT_PASSO_PRED-TAM_JANELA)

for i in range(QT_PASSO_PRED):
    #Monta a primeira linha de dados de entrada para serem preditos.
    #Pega os dados da ultima linha de treino e junta com o resultado da ultima linha de treino
    if(i==0):
        
#        ultimo_y_real = df_pontos_y.tail(1)
        
#        arr_tmp = df_pontos_x.tail(1).values
        primeiralinha = df_treino.tail(1)

        ultimo_y_real = y_treino.tail(1)
        
        arr_tmp = df_treino.tail(1)
        
        print("Adiciona o ultimo valor (rotulo) dos dados de treino: (%f) " % ultimo_y_real)
        arr_tmp = np.append(arr_tmp, ultimo_y_real)
        print("Faz o shift deletando o primeiro valor da coluna valor: (%f)" % arr_tmp[0])
        arr_tmp = np.delete(arr_tmp, 0)
        
        df_tmp = pd.Series(arr_tmp).to_frame().transpose()
        df_tmp.columns = features.columns
        df_teste = df_teste.append(df_tmp, ignore_index=True)
        
    dado_to_predict = df_teste.tail(1)
    
    valor_predito = rna.predict(dado_to_predict)
#    print("Valor (%d) Predito: " % i)
#    print(valor_predito)
    
#    print("Pega a última linha dos dados de treino e transforma em um array: ")
    arr_tmp = df_teste.tail(1).values
    
#    print("Adiciona o ultimo valor (rotulo) predito: (%f) " % valor_predito)
    arr_tmp = np.append(arr_tmp, valor_predito)
    
#    print("Faz o shift deletando o primeiro valor da linha de teste: (%f)" % arr_tmp[0])
    arr_tmp = np.delete(arr_tmp, 0)
    df_tmp = pd.Series(arr_tmp).to_frame().transpose()
    df_tmp.columns = colunas
    df_teste = df_teste.append(df_tmp, ignore_index=True)
    
#    print("Guarda em uma lista para poder mensurar o modelo:")
    predicoes_passos.append(valor_predito)

#df_teste = df_teste.drop(df_teste.index[0])
    
plota_grafico_2(labels.tail(QT_PASSO_PRED),
          labels.tail(QT_PASSO_PRED), 
          predicoes_passos, 
          "Predição de " + str(QT_PASSO_PRED) + " passo(s) \n com deslocamento temporal", 
          QT_PASSO_PRED)

metricas(labels.tail(QT_PASSO_PRED), predicoes_passos)