import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

dirname = os.path.dirname(__file__)

def visualizar_reta(X, y, theta):
    t = np.arange(0, 25, 1)
    plt.scatter(X, y, color='red', marker='x', label='Training Data')
    plt.plot(t, theta[0] + (theta[1]*t), color='blue', label='Linear Regression')
    plt.axis([4, 25, -5, 25])
    plt.title('População da cidade x Lucro da filial')
    plt.xlabel('População da cidade (10k)')
    plt.ylabel('Lucro (10k)')
    plt.legend()
    plt.savefig(dirname + os.path.sep + 'plot1.2.png')
    plt.show()
