import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def get_features_and_targets(filepath, add_ones = True):
    path = os.path.join(os.path.dirname(__file__), filepath)  
    data = pd.read_csv(path, header=None)

    # adiciona uma coluna de 1s referente a variavel x0
    if (add_ones):
        data.insert(0, 'Ones', 1)
    
    # separa os conjuntos de dados x (caracteristicas) e y (alvo)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols]
    
    # converte os valores em numpy arrays
    X = np.array(X.values)  
    y = np.array(y.values)
    
    return X,y