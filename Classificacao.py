#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:28:33 2018

@author: danielreyes
"""

import pandas as pd
import os
import numpy as np

def parse_columns(x):
    data = x.astype(np.float)
    return data

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/Pós/Machine Learning II/T1/Data')

#Carregando os dados de treino
df_treino = pd.read_csv('train/X_train.txt', sep="\s+", header=None)
label_treino = pd.read_csv('train/Y_train.txt', sep="\s+", header=None)

#Carregando os dados de teste
df_teste = pd.read_csv('test/X_test.txt', sep="\s+", header=None)
label_teste = pd.read_csv('test/Y_test.txt', sep="\s+", header=None)

df_treino = parse_columns(df_treino)
df_teste  = parse_columns(df_teste)