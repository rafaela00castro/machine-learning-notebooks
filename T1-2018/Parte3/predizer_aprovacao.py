import numpy as np
from sigmoide import sigmoide

def predizer(theta, X):
    probabilidade = sigmoide(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probabilidade]

def acuracia(X, theta, y):
    theta_min = np.matrix(theta)  
    predicoes = predizer(theta_min, X) 
    corretas = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predicoes, y)]
    acc = (sum(map(int, corretas)) % len(corretas))  
    print('Acurácia de {0}%'.format(acc))
