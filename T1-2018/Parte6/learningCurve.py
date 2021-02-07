import numpy as np
import matplotlib.pyplot as plt
import os
from Parte5 import linearRegCostFunction as cfunc

def learningCurve(theta, X, y, Xval, yval, _lambda):
    m = len(X)
    erros_treino = np.zeros(m)
    erros_val = np.zeros(m)
    numero_exemplos = []
    
    for i in range(1,m+1):
        treino_subset = X[:i,:]
        y_subset = y[:i]
        numero_exemplos.append(len(treino_subset))
        
        result = cfunc.encontrar_theta_otimo(theta, treino_subset, y_subset, _lambda)
        theta = result[0]
        
        J_treino = cfunc.custo_reglin_regularizada(theta, treino_subset, y_subset, _lambda=0)
        J_val = cfunc.custo_reglin_regularizada(theta, Xval, yval, _lambda)
        
        erros_treino[i-1] = J_treino
        erros_val[i-1] = J_val
    
    return numero_exemplos, erros_treino, erros_val
    
def plot_learning_curve(x,y_treino, y_val, nome_arquivo, titulo=None):
    plt.figure(figsize=(8,5))
    plt.plot(x,y_treino,label='Treinamento')
    plt.plot(x,y_val,label='Validação cruzada')
    plt.title('Curva de aprendizado para regressão linear')
    plt.xlabel('Número de exemplos treinados')
    plt.ylabel('Erro')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,y2))
    plt.legend()
    plt.title(titulo)
    
    plt.savefig(os.getcwd() + os.path.sep + nome_arquivo)
    plt.show()