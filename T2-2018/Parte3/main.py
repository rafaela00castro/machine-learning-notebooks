import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlab
from scipy.io import loadmat  
from scipy import stats 
import argparse as arg
import warnings

# inserido para evitar o aviso MatplotlibDeprecationWarning: The bivariate_normal function was deprecated in version 2.2.
warnings.filterwarnings("ignore")

def estimate_gaussian_params(X):  
	mu = (np.sum(X, axis=0)) / X.shape[0] # = np.mean(X, axis=0) or X.mean(axis=0)
	sigma2 = (np.sum((X - mu)**2, axis=0)) / X.shape[0] # = np.var(X, axis=0) or X.var(axis=0)

	return (mu, sigma2)

def select_epsilon(pval, yval):  
	best_epsilon_value = 0
	best_f1_value = 0

	step_size = (pval.max() - pval.min()) / 1000

	print('step size: ' + str(step_size))

	for epsilon in np.arange(pval.min(), pval.max(), step_size):
		preds = pval < epsilon
		tp = np.sum(np.logical_and(preds == 1, yval == 1)) #verdadeiro positivo
		fp = np.sum(np.logical_and(preds == 1, yval == 0)) #falso positivo
		fn = np.sum(np.logical_and(preds == 0, yval == 1)) #falso negativo

		prec = tp / (tp + fp)
		rec = tp / (tp + fn)
		f1 = (prec * rec * 2) / (prec + rec)

		if f1 > best_f1_value:
			best_epsilon_value = epsilon
			best_f1_value = f1

	return best_epsilon_value, best_f1_value

def get_arguments():
	parser = arg.ArgumentParser()
	parser.add_argument('-f', '--only-first-plot', action='store', dest='only_first_plot', default=False)
	parser.add_argument('-s', '--only-second-plot', action='store', dest='only_second_plot', default=False)

	return parser.parse_args()

def main():
	data = loadmat('./data/ex8data1.mat') 
	X = data['X']
	
	args = get_arguments()
	if (args.only_first_plot):
		# Plot dataset
		plt.scatter(X[:,0], X[:,1], marker='x')  
		plt.axis('equal')
		plt.xlabel('Latência (ms)')
		plt.ylabel('Vazão (mb/s)')
		plt.title('Conjunto de dados - servidores em um data center') 
		plt.show()
		return
		
	(mu, sigma2) = estimate_gaussian_params(X)

	if (args.only_second_plot):
		print('mu: ' + str(mu))
		print('variance: ' + str(sigma2) + '\n ')
		# Plot dataset and contour lines
		plt.scatter(X[:,0], X[:,1], marker='x')  
		x = np.arange(0, 25, .025)
		y = np.arange(0, 25, .025)
		plt.xlabel('Latência (ms)')
		plt.ylabel('Vazão (mb/s)')
		first_axis, second_axis = np.meshgrid(x, y)
		Z = mlab.bivariate_normal(first_axis, second_axis, np.sqrt(sigma2[0]), np.sqrt(sigma2[1]), mu[0], mu[1])
		plt.contour(first_axis, second_axis, Z, 10, cmap=plt.cm.jet)
		plt.axis('equal')
		plt.title('Conjunto de dados - servidores em um data center')
		plt.show()
		return
		
	# Load validation dataset
	Xval = data['Xval']  
	yval = data['yval'].flatten()

	stddev = np.sqrt(sigma2)

	pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
	pval[:,0] = stats.norm.pdf(Xval[:,0], mu[0], stddev[0])  
	pval[:,1] = stats.norm.pdf(Xval[:,1], mu[1], stddev[1])  
	print(np.prod(pval, axis=1).shape)
	epsilon, _ = select_epsilon(np.prod(pval, axis=1), yval)  
	print('Best value found for epsilon: ' + str(epsilon) + '\n ')

	# Computando a densidade de probabilidade 
	# de cada um dos valores do dataset em 
	# relação a distribuição gaussiana
	p = np.zeros((X.shape[0], X.shape[1]))  
	p[:,0] = stats.norm.pdf(X[:,0], mu[0], stddev[0])  
	p[:,1] = stats.norm.pdf(X[:,1], mu[1], stddev[1])

	# Apply model to detect abnormal examples in X
	anomalies = np.where(np.prod(p, axis=1) < epsilon)

	# Plot the dataset X again, this time highlighting the abnormal examples.
	plt.clf()
	plt.scatter(X[:,0], X[:,1], marker='x', label='Normais')  
	plt.scatter(X[anomalies[0],0], X[anomalies[0],1], s=50, color='r', marker='x', label='Anomalias')  
	plt.axis('equal')
	plt.xlabel('Latência (ms)')
	plt.ylabel('Vazão (mb/s)')
	plt.title('Conjunto de dados - servidores em um data center - anomalias destacadas')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()