import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

# Carregando os dados do dataset e armazendo em um array. Em seguida damos uma rápida visualizada nos dados
data = pd.read_csv('Parte3/ex2data1.txt', header=None, names=['Prova 1', 'Prova 2', 'Aprovado'])  
data.head() 

# converte de dataframes para arrays
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# converte de arrays para matrizes
X = np.array(X.values)  
y = np.array(y.values)  

# gerando o gráfico de dispersãara análise preliminar dos dados
positivo = data[data['Aprovado'].isin([1])]  
negativo = data[data['Aprovado'].isin([0])]

fig, ax = plt.subplots(figsize=(8,5))  
ax.scatter(positivo['Prova 1'], positivo['Prova 2'], s=50, c='k', marker='+', label='Aprovado')  
ax.scatter(negativo['Prova 1'], negativo['Prova 2'], s=50, c='y', marker='o', label='Não Aprovado')  
ax.legend()  
ax.set_xlabel('Nota da Prova 1')  
ax.set_ylabel('Nota da Prova 2')  