# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Imports

# <codecell>

import warnings
warnings.filterwarnings('ignore')
# biblioteca para sistema operacional
import os
# biblioteca para plots
from matplotlib import pyplot as pl
# biblioteca para dados estruturados
import pandas as pd
# biblioteca para manipulação de arrays
import numpy as np
# função para geração de arrays cíclicos
from itertools import cycle
# random
import random
import scipy.io
from itertools import repeat

# <headingcell level=1>

# Constantes

# <codecell>

# pasta com os dados
PATH_DADOS = "Dados Exercicio 3"

# constantes da questão 1

Q1_FILEPATH = os.path.join(PATH_DADOS, "ex3data1.mat")
Q2_FILEPATH = os.path.join(PATH_DADOS, "ex3data2.data")

# <headingcell level=1>

# Funções

# <codecell>

# recebe um vetor x de dimensão (n, d, d) e um vetor y de dimensão (n, c)
def visualizar_imgs(X, T):
    
    n = X.shape[0]
    
    d_n = floor(sqrt(n))+1
    
    fig = pl.figure(figsize=(d_n,d_n))
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in xrange(n):
        
        ax = fig.add_subplot(d_n, d_n, i+1, xticks=[], yticks=[])
        ax.imshow(X[i], cmap=pl.cm.binary)
        
        ax.text(0, 18, str(T[i].argmax()))
        
def output_rede(hidden, output, x):

    if x.ndim == 1:
        x = x.reshape((x.shape[0], 1))
        
    x = np.vstack((-1*np.ones((1, x.shape[1])), x))
    
    y_hidden = np.dot(hidden, x)
    sigma_hidden = 1./(1. + np.exp(-1.*y_hidden))
    
    sigma_hidden = np.vstack((-1*np.ones((1, sigma_hidden.shape[1])), sigma_hidden))
   
    y_output = np.dot(output, sigma_hidden)
    sigma_output = 1./(1. + np.exp(-1.*y_output))
    
    return sigma_output, sigma_hidden

def eqm(esperados, retornados):
    
    if esperados.ndim > 1:
        esperados = np.apply_along_axis(lambda x: x.argmax(), 1, esperados)
        retornados = np.apply_along_axis(lambda x: x.argmax(), 1, retornados)
        
#     print "e:", esperados
#     print "r:", retornados
    
    return .5*np.sum((esperados - retornados)**2)    

def indices_random(ns):
    
    n = np.sum(ns)
    i_ns = np.cumsum(ns)
    
    indices_random = np.random.permutation(np.arange(0, n, 1))
    
    return np.array_split(indices_random, i_ns)[:-1]    

# <headingcell level=1>

# Questão 1

# <headingcell level=3>

# Carregamento dos dados

# <codecell>

dados_mat = scipy.io.loadmat(Q1_FILEPATH)

# <codecell>

T = np.array(dados_mat['T'])
X = np.array(dados_mat['X'])

# <headingcell level=3>

# Pesos

# <headingcell level=1>

# Gradiente Descendente

# <codecell>

def gradiente_descendente(X, T, n_hidden, lamb=.01, w_inicial=.05, n_treino=4000, n_validacao=500, n_teste=500, max_epocas=100):
    
    # cálculo do número de exemplos
    n = X.shape[0]
    
    # cálculo do tamanho de entrada e saída
    n_input = X.shape[1]
    n_output = T.shape[1]
    
    # erro de validação inicial - utilizado como condição de parada
    eqm_validacao_anterior = 2**20
    eqm_validacao = 2**20 - 1
    
    # inicialização da rede
    hidden = w_inicial*np.ones((n_hidden, n_input + 1))
    output = w_inicial*np.ones((n_output, n_hidden + 1))
    
    # partição em conjuntos de treinamento, validação e teste
    indices_treino, indices_validacao, indices_teste = indices_random([n_treino, n_validacao, n_teste])
    
    X_treino = np.take(X, indices_treino, axis=0)
    X_validacao = np.take(X, indices_validacao, axis=0)
    X_teste = np.take(X, indices_teste, axis=0)
    
    T_treino = np.take(T, indices_treino, axis=0)
    T_validacao = np.take(T, indices_validacao, axis=0)
    T_teste = np.take(T, indices_teste, axis=0)
    
    eqms_validacao = []
    eqms_treino = []
    
    while eqm_validacao < eqm_validacao_anterior and len(eqms_validacao) < max_epocas:
        
        # TODO: randomizar o X e o T
        for i in xrange(n_treino):
            
            x = X_treino[i]
            x = x.reshape((x.shape[0], 1))
            t = T_treino[i]
            t = t.reshape((t.shape[0], 1))
            
            # execução da rede

            X = np.vstack((-1*np.ones((1,1)), x))

            y_hidden = np.dot(hidden, X)
            sigma_hidden = 1./(1. + np.exp(-1.*y_hidden))

            Y = np.vstack((-1*np.ones((1,1)), sigma_hidden))
           
            y_output = np.dot(output, Y)
            sigma_output = 1./(1. + np.exp(-1.*y_output))
            
            import ipdb; ipdb.set_trace()
            # erro da camada output
            e_output = sigma_output*(1. - sigma_output)*(t - sigma_output)
            
            # erro na camada hidden
            e_hidden = sigma_hidden*(1. - sigma_hidden)*np.dot(output[:, 1:].T, e_output)
            
            # atualização dos pesos

            hidden = hidden + lamb*np.dot(e_hidden, X.T)

            output = output + lamb*np.dot(e_output, Y.T)
            
        # cálculo dos erros pros conjuntos
        y_treino, _ = output_rede(hidden, output, X_treino.T)
        eqm_treino = eqm(T_treino, y_treino.T)
        eqms_treino.append([eqm_treino, hidden, output])
        
        y_validacao, _ = output_rede(hidden, output, X_validacao.T)
        eqm_validacao_anterior = eqm_validacao
        eqm_validacao = eqm(T_validacao, y_validacao.T)
        eqms_validacao.append(eqm_validacao)
        
    return hidden, output, eqms_treino, eqms_validacao

# <codecell>

h, o, t, v = gradiente_descendente(X, T, 200, w_inicial=.5, max_epocas=100, lamb=.75)

