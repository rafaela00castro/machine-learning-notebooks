import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

dirname = os.path.dirname(__file__)
filepath = os.path.sep + 'ex1data1.txt'

def importarDados(filepath,names):
    #path = os.getcwd() + filepath
    path = dirname + filepath 
    data = pd.read_csv(path, header=None, names=names)
    
    # separa os conjuntos de dados x (caracteristicas) e y (alvo)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols]
    
    # converte os valores em numpy arrays
    X = np.array(X.values)  
    y = np.array(y.values)
    
    return X,y

population, profit = importarDados(filepath,["Population","Profit"])

plt.scatter(population, profit, color='red', marker='x')
plt.title('População da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.savefig(dirname + os.path.sep + 'plot1.1.png')
plt.show()
