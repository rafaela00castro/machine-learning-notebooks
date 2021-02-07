import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt
import argparse as arg

def find_closest_centroids(X, centroids):    
	# necessário para fazer o broadcasting
	centroids_extended = centroids[: , np.newaxis, :]

	distance = np.sum(((X - centroids_extended)**2), axis=2)
	idx = np.argmin(distance, axis=0)
	return idx

def compute_centroids(X, idx, K):
	centroids = [X[idx==i].mean(axis=0) for i in range(K)]
	return np.array(centroids)

def kmeans_init_centroids(X, K):
	return X[np.random.choice(X.shape[0], K, replace=False)]

def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
	K = np.size(initial_centroids, 0)
	centroids = initial_centroids 
	previous_centroids = centroids

	for iter in range(max_iters):
		# Assignment of examples do centroids
		idx = find_closest_centroids(X, centroids)

		# PLot the evolution in centroids through the iterations
		if plot_progress:
			plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
			plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
			plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
			plt.plot(previous_centroids[:,0], previous_centroids[:,1], 'yo')
			plt.plot(centroids[:,0], centroids[:,1], 'bo')
			plt.title('Iteration #' + str(iter+1))
			plt.show()

		previous_centroids = centroids

		# Compute new centroids
		centroids = compute_centroids(X, idx, K)

	return (centroids, idx)

def get_arguments():
	parser = arg.ArgumentParser()
	parser.add_argument('-p', '--plot-progress', action='store', dest='plot_progress', default=False)
	parser.add_argument('-r', '--random-init', action='store', dest='random_init', default=False)
	parser.add_argument('-o', '--only-find-centroid', action='store', dest='only_find_centroid', default=False)

	return parser.parse_args()


def main():
	# Find closest centroids
	raw_mat = scipy.io.loadmat("./data/ex7data2.mat")
	X = raw_mat.get("X")
	K = 3

	args = get_arguments()
	if(args.random_init):
		# random seeds
		initial_centroids = kmeans_init_centroids(X, K)
		print('Initial centroids:\n', initial_centroids) 

	else:
		# Fixed seeds (i.e., initial centroids)
		initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
		
	idx = find_closest_centroids(X, initial_centroids)
	
	# Plot initial assignments.
	plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
	plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
	plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
	plt.title('Initial assignments')
	plt.show()
	
	print('Cluster assignments for the first, second and third examples: ' + str(idx[0:3].flatten()))

	centroids = initial_centroids
	if (not args.only_find_centroid):
		# Compute initial means
		centroids = compute_centroids(X, idx, K)
		# Now run 10 iterations of K-means on fixed seeds	
		max_iters = 10
		centroids, idx = run_kmeans(X, initial_centroids, max_iters, args.plot_progress)

	print('Centroids after the 1st update:\n' + str(centroids))

	# Plot final clustering.
	plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
	plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
	plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
	plt.title('Final clustering')
	plt.show()

if __name__ == "__main__":
	main()
