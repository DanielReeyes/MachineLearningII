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
QT_PASSO_PRED = 100
TAM_JANELA = 15

#Função para criar um data set e posteriormente dividi-lo em treino e teste
#Dentro da função há a equação que produzirá os valores
def monta_dataset(RANGE_N):
    pontos = []
    valores_n = []    
    n_passos = pd.Series(np.arange(0.0, RANGE_N, TAM_PASSO))
    for n in n_passos:
        result = math.sqrt(1.0 + (math.sin(n + (math.sin(n) ** 2.0))))
        pontos.append(float(result))
        valores_n.append(float(n))
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

def plota_grafico_1passo(y_real, y_predito, titulo):
    y_real = np.array(y_real)    
    y_predito = np.array(y_predito)
    
    objects = ('Real', 'Predito')
    x_pos = np.arange(len(objects))
    performance = [y_real,y_predito]
    color = ['green', 'red']

    plt.bar(x_pos, performance, align='center', alpha=0.5, color=color)
    plt.xticks(x_pos, objects, fontsize=12, rotation=30)
    plt.xlabel('Valor do próximo passo', fontsize=12)
    plt.title(titulo)
     
    plt.show()

X, y = monta_dataset(RANGE_N)
#df_y = pd.Series(y)
#df_x = pd.Series(X)
#df_y.plot()


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
rna = MLPRegressor(hidden_layer_sizes=(100, 100, 100,100), 
                   activation='tanh', 
                   solver='adam',
                   early_stopping=True)

#Cria um dataframe apenas com as variáveis, excluindo os rótulos do momento 'x' a ser predito posteriormente
features = df_pontos.drop('x', axis=1)

#Cria um dataframe\lista com os rótulos reais
labels = df_pontos['x']

#Declara um scaler, para normalizar os dados entre -1 e 1
max_abs_scaler = MaxAbsScaler()
features = max_abs_scaler.fit_transform(features)
features = DataFrame(features)
#max_abs_scaler = max_abs_scaler.fit(features)

#treina o modelo com o dataframe de variáveis e os rótulos
rna.fit(features, labels)

#Simula um dataset de teste, porém com os dados de treinos, pegando os últimos X dados que seriam testados
features_to_test_model = features.tail(QT_PASSO_PRED)
predicoes = rna.predict(features_to_test_model)

if(QT_PASSO_PRED==1):
    plota_grafico_1passo(labels.tail(QT_PASSO_PRED),
                         predicoes,
                         "Predição de " + str(QT_PASSO_PRED) + " passo(s) \n com deslocamento temporal")
else:    
    plota_grafico_2(labels.tail(QT_PASSO_PRED),
                  labels.tail(QT_PASSO_PRED), 
                  predicoes, 
                  "Predição de " + str(QT_PASSO_PRED) + " passo(s) \n sem deslocamento temporal. Fase: Treino", 
                  QT_PASSO_PRED)

#Agora mensura o quanto o modelo acertou com os dados de treino
metricas(labels.tail(QT_PASSO_PRED), predicoes)

#Criando uma lista pra guardar os valores preditos para cada passo
predicoes_passos = []
#Criando um dataset para guardar os valores que foram utilizados para predizer os próximos N passos
df_teste = DataFrame()

#Monta o dataset de treino para os testes
#Pega o dataset original, menos a quantidade de dados pra predição, menos a quantidade de dados da janela
df_treino = features.head(len(X) - QT_PASSO_PRED-TAM_JANELA)
y_treino = labels.head(len(X) - QT_PASSO_PRED-TAM_JANELA)

for i in range(QT_PASSO_PRED):
    #Monta a primeira linha de dados de entrada para serem preditos.
    #Pega os dados da ultima linha de treino e junta com o resultado da ultima linha de treino
    if(i==0):

#        Vamos guardar os dados que foram usados como dataset de teste
#        Como ele está vazio, caso seja a primeira linha, irá pegar a ultima linha de dados
#        do dataset de treino e o seu rótulo
#        Transformo em array só para poder manuseá-lo
        arr_tmp = df_treino.tail(1)
        ultimo_y_real = y_treino.tail(1)        

#       Valor a ser predito é o instante x
#       Dessa forma, adiciono o ultimo valor a linha como se fosse o instante x-1
        print("Adiciona o ultimo valor (rotulo) dos dados de treino: (%f) " % ultimo_y_real)
        arr_tmp = np.append(arr_tmp, ultimo_y_real)

#       Agora removo o valor do último instante que conhecemos
        print("Faz o shift deletando o primeiro valor da coluna valor: (%f)" % arr_tmp[0])
        arr_tmp = np.delete(arr_tmp, 0)

#       Transforma em uma linha de dataframe pra poder concatenar no dataframe de teste
        df_tmp = pd.Series(arr_tmp).to_frame().transpose()
        df_tmp.columns = features.columns
        df_tmp = pd.DataFrame(max_abs_scaler.transform(df_tmp))
        df_teste = df_teste.append(df_tmp, ignore_index=True)

#   Até agora não houve nenhuma predição dos dados de teste
#   Pegamos a última linha do dataframe de teste pois é o próximo passo que deve ser predito
    dado_to_predict = df_teste.tail(1)

#   Prediz o próximo passo da série temporal    
    valor_predito = rna.predict(dado_to_predict)
#    print("Valor (%d) Predito: " % i)
#    print(valor_predito)

#   Pegamos a última linha de teste (a mesma usada para predição) para poder manuseá-la    
#    print("Pega a última linha dos dados de treino e transforma em um array: ")
    arr_tmp = df_teste.tail(1).values

#   Adiciona o valor predito à linha    
#    print("Adiciona o ultimo valor (rotulo) predito: (%f) " % valor_predito)
    arr_tmp = np.append(arr_tmp, valor_predito)
    
#   Faz o shift, deletando o valor do instante mais antigo que se conhece da linha
#    print("Faz o shift deletando o primeiro valor da linha de teste: (%f)" % arr_tmp[0])
    arr_tmp = np.delete(arr_tmp, 0)

#   Transforma o array em uma linha de dataframe para poder concatenar ao dataframe de teste
    df_tmp = pd.Series(arr_tmp).to_frame().transpose()
    df_tmp.columns = features.columns
    df_tmp = pd.DataFrame(max_abs_scaler.transform(df_tmp))
    df_teste = df_teste.append(df_tmp, ignore_index=True)
    
#   Guarda em uma lista para poder mensurar o modelo:
#    print("Guarda em uma lista para poder mensurar o modelo:")
    predicoes_passos.append(valor_predito)

#df_teste = df_teste.drop(df_teste.index[0])
    
#plota_grafico_2(labels.tail(QT_PASSO_PRED),
if(QT_PASSO_PRED == 1):
    plota_grafico_1passo(labels.tail(QT_PASSO_PRED), 
                         predicoes_passos, 
                         "Predição de " + str(QT_PASSO_PRED) + " passo(s) \n com deslocamento temporal")
else:
    plota_grafico_2(y,
          labels.tail(QT_PASSO_PRED), 
          predicoes_passos, 
          "Predição de " + str(QT_PASSO_PRED) + " passo(s) \n com deslocamento temporal", 
          QT_PASSO_PRED)

metricas(labels.tail(QT_PASSO_PRED), predicoes_passos)