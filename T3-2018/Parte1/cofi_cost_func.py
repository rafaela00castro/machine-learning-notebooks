import numpy as np


def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda):

	# Obtém as matrizes X e Theta a partir dos params
	X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
	Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()
	
	# Você deve retornar os seguintes valores corretamente
	J = 0
	X_grad = np.zeros(X.shape)
	Theta_grad = np.zeros(Theta.shape)

	H = X.dot(Theta.T)
	error = np.multiply(H - Y, R)
	J = (np.sum(np.power(error, 2))) / 2
	
	# custo com regularização
	J = J + ((Lambda /2) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2))))

	# gradiente com regularização
	X_grad = error.dot(Theta) + (Lambda * X)
	Theta_grad = error.T.dot(X) + (Lambda * Theta)
	
	grad = np.hstack((X_grad.T.flatten(),Theta_grad.T.flatten()))

	return J, grad