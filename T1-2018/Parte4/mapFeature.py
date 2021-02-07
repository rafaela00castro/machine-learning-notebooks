import pandas as pd
import numpy as np

def mapFeature(x1, x2, grau):     
    df = pd.DataFrame(index=range(x1.shape[0]))
    
    for i in range(1, grau+1):
        for j in range(i+1):
            df['x_' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
    df.insert(0, 'x_0', 1)
    
    return df.values   
    