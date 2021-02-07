import numpy as np
import scipy.optimize as opt

def custo_reglin_regularizada(theta, X, y, _lambda):
    # Quantidade de exemplos
    m = len(X)
    theta = np.matrix(theta)

    # não considera theta0 para o cálculo
    theta_j = theta[:,1:]
    regularizacao = (_lambda /(2 * m)) * np.sum(theta_j.dot(theta_j.T))    

    erro = X.dot(theta.T) - y

    # Computa a função de custo J
    J = (np.sum(np.power(erro, 2)))/ (2 * m) 
    
    return J + regularizacao


def gd_regularizada(theta, X, y, _lambda):
    m = len(X)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    erro = (X.dot(theta.T)) - y
    
    gradient = X.T.dot(erro) / m

    theta_j = theta[:,1:]
    regularizacao = (_lambda / m) * theta_j
    # insere zero como termo de regularização para theta0
    regularizacao = np.insert(regularizacao, 0, 0, axis=1)

    return gradient + regularizacao.T


def encontrar_theta_otimo(theta, X, y, _lambda):
    return opt.fmin_tnc(func=custo_reglin_regularizada, x0=theta, fprime=gd_regularizada, args=(X, y, _lambda))


    