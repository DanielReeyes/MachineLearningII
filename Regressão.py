#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:19:06 2018

@author: danielreyes
"""

import math
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


range_n = 10.0
tam_passo = 0.01

def monta_dataset(range_n, tam_passo):
    pontos = []
    n_passos = int(range_n / tam_passo)
    print("Quantidade de passos série temporal: ")
    print(n_passos)
    for n in range(n_passos):
        result = math.sqrt(1 + (math.sin(n + (math.sin(n) ** 2))))
        pontos.append(result)
    return pontos
        

df_y = monta_dataset(range_n + tam_passo, tam_passo)
df_y = pd.Series(df_y)

df_x = pd.Series(range(1001))

df = df_x.to_frame('Passo')
df['Valor'] = df_y.to_frame()
#df_y.plot()

X_treino = df['Passo'].to_frame()
y_treino = df['Valor'].to_frame()

qtd_linhas_treino = X_treino['Passo'].count()

X_teste = list(range(qtd_linhas_treino, (qtd_linhas_treino + 100)))
X_teste = pd.DataFrame(X_teste)

y_real = monta_dataset(11.0, 0.01)
y_real = pd.Series(y_real)
y_real = pd.DataFrame(y_real)
y_real = y_real.tail(100)

rna = MLPRegressor()
print(y_treino.values.ravel())

rna = rna.fit(X=X_treino, y=y_treino.values.ravel())
predicoes = rna.predict(X_teste)

#predicoes.plot()
pred = pd.Series(predicoes)
pred.plot()

r2_error = r2_score(y_true=y_real, y_pred=predicoes, multioutput="uniform_average")
print("Erro.: ")
print(r2_error)