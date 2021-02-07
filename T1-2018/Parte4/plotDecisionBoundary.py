import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from sigmoide import sigmoide
from mapFeature import mapFeature

def plot_boundary(theta, grau):
    x = np.linspace(-1,1.5,50)
    y = np.linspace(-0.8,1.2,50)
    
    xx, yy = np.meshgrid(x, y)

    theta = np.matrix(theta)
    
    X_poly = mapFeature(xx.ravel(), yy.ravel(), grau)
    
    Z = sigmoide(X_poly.dot(theta.T))
    Z = Z.reshape(xx.shape)
    
    plt.title('lambda = 1')
    plt.contour(x, y, Z, [0.5], linewidths=1, colors='green')
    
    legendas = [Line2D([0], [0], marker='+', color='k', lw=0, label='Aceito (y = 1)'),
                       Line2D([0], [0], marker='o',color='y', lw=0, label='Rejeitado (y = 0)'),
                       Line2D([0], [0], color='g', lw=2, label='Fronteira de Decisão')]
    
    plt.legend(handles=legendas)
    
    dirname = os.path.dirname(__file__)
    plt.savefig(dirname + os.path.sep + '/plot4.2.png')
    plt.show()