import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dirname = os.path.dirname(__file__)

def plot_surface(J):
    # Valores de theta0 e theta1 informados no enunciado do trabalho
    theta0 = np.arange(-10, 10, 0.01)
    theta1 = np.arange(-1, 4, 0.01)

    # Comandos necessários para o matplotlib plotar em 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plotando o gráfico de superficie
    theta0, theta1 = np.meshgrid(theta0, theta1) 
    surf = ax.plot_surface(theta0, theta1, J)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.savefig(dirname + os.path.sep + 'plot1.4.png')
    plt.show()