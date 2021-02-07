import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.io
import argparse as arg

def normalize_features(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0)
	normalized_X = np.divide(X - mu,sigma)

	return (normalized_X, mu, sigma)

def pca(X):
	cov_matrix = (X.T.dot(X)) / X.shape[0] # = np.cov(X, axis=0) or X.cov(axis=0)
	
	U, S, V = np.linalg.svd(cov_matrix)
	return (U, S)

def project_data(X, U, K):
	U_reduce = U[:, 0:K]
	Z = np.zeros((len(X), K))
	for i in range(len(X)):
		x = X[i,:]
		projection_k = np.dot(x, U_reduce)
		Z[i] = projection_k
	return Z

def recover_data(Z, U, K):
	X_rec = np.zeros((len(Z), len(U)))
	for i in range(len(Z)):
		v = Z[i,:]
		for j in range(np.size(U,1)):
			recovered_j = np.dot(v.T,U[j,0:K])
			X_rec[i][j] = recovered_j
	return X_rec

def explain_variance(S):
	total = np.sum(S)
	variance_percent = (S / total) * 100
	cumulative_variance_percent = np.cumsum(variance_percent)
	
	bars = ['PC1','PC2']
	plt.bar(bars, variance_percent)
	plt.plot(bars, cumulative_variance_percent, marker='o', color='g', label='variância acumulada')
	plt.title('Variância por diferentes componentes principais (PC)')
	plt.ylabel('Variância em porcentagem')
	plt.legend()
	plt.show()
	
def get_arguments():
	parser = arg.ArgumentParser()
	parser.add_argument('-o', '--only-first-plot', action='store', dest='only_first_plot', default=False)
	parser.add_argument('-n', '--no-plot', action='store', dest='no_plot', default=False)

	return parser.parse_args()

def main():
	raw_mat = scipy.io.loadmat("./data/ex7data1.mat")
	X = raw_mat.get("X")
	Z, X_rec = 0, 0
	
	args = get_arguments()
	if (args.only_first_plot):
		plt.cla()
		plt.plot(X[:,0], X[:,1], 'bo')
		plt.title('Conjunto de dados original')
		plt.show()
		return Z, X_rec

	X_norm, mu, sigma = normalize_features(X)
	U, S = pca(X_norm)

	if (not args.no_plot):
		plt.cla()
		plt.axis('equal')
		plt.plot(X_norm[:,0], X_norm[:,1], 'bo')

		K = 2
		for axis, color in zip(U[:K], ["yellow","green"]):
			start, end = np.zeros(2), (mu + sigma * axis)[:K] - (mu)[:K]
			plt.annotate('', xy=end,xytext=start, arrowprops=dict(facecolor=color, width=1.0))
		plt.axis('equal')
		plt.title('Conjunto de dados original com autovetores')
		plt.show()

	K = 1
	Z = project_data(X_norm, U, K)
	X_rec = recover_data(Z, U, K)

	if (not args.no_plot):
		plt.cla()
		
		explain_variance(S)
		
		plt.plot(X_norm[:,0], X_norm[:,1], 'bo', label='Pontos de dados originais')
		plt.plot(X_rec[:,0], X_rec[:,1], 'rx', label='Pontos de dados projetados')
		plt.legend()
		plt.axis('equal')
		plt.title('Conjuntos de dados original e projetado')
		plt.show()
	
	return Z, X_rec

if __name__ == "__main__":
	Z, X_rec = main()
