import numpy as np

np.seterr(over='ignore')

def sigmoide(z):
    return 1.0 / (1 + np.exp(-z))
