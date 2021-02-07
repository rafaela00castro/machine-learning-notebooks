import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import os

# Carregando os dados do dataset e armazendo em um array. Em seguida damos uma rápida visualizada nos dados
data = pd.read_csv('Parte4/ex2data2.txt', header=None, names=['Teste 1', 'Teste 2', 'Aceito'])  
data.head() 

# converte de dataframes para arrays
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# converte de arrays para matrizes
X = np.array(X.values)  
y = np.array(y.values)  

# gerando o gráfico de dispersão para análise preliminar dos dados
positivo = data[data['Aceito'].isin([1])]  
negativo = data[data['Aceito'].isin([0])]

fig, ax = plt.subplots(figsize=(8,5))  
ax.scatter(positivo['Teste 1'], positivo['Teste 2'], s=50, c='k', marker='+', label='Aceito (y = 1)')  
ax.scatter(negativo['Teste 1'], negativo['Teste 2'], s=50, c='y', marker='o', label='Rejeitado (y = 0)')  
ax.legend()  
ax.set_xlabel('Microchip Teste 1')  
ax.set_ylabel('Microchip Teste 2')

dirname = os.path.dirname(__file__)
fig.savefig(dirname + os.path.sep + '/plot4.1.png')