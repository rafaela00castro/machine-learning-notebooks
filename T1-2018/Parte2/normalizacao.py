import numpy as np

def normalizar_caracteristica(X, y=0):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_norm = (X - mean_X) / std_X
	
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)
    y_norm = (y - mean_y) / std_y
	
    return X_norm, y_norm, mean_X, std_X, mean_y, std_y

