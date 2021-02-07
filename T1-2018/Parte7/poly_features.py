import pandas as pd
import numpy as np

def poly_features(X, p):
    df = pd.DataFrame(index=range(X.shape[0]))
    
    for i in range(1,p+1):
        df['x_' + str(i)] = np.power(X, i)
    
    X_poli = df.values 
    return X_poli

    