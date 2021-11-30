import numpy as np
from numpy import random
from numpy.lib.function_base import gradient

# X es el dataset
# y son las etiquetas
# eta es la velocidad de aprendizaje
# n_iter es el numero de iteraciones
def BGD(X, y, eta=0.01, n_iter=100):
    m, features = X.shape
    X_ = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(features+1, 1)

    for iteration in range(n_iter):
        gradient = 2/m * X_.T @ (X_ @ theta - y)
        theta = theta - eta*gradient

    return theta

# t0 t1 hiperparámetros de learning schedule
def learning_schedule(t, t0, t1):
    return t0 / (t + t1)

def SGD(X, y, n_epochs=100, t0=1, t1=500):

    m, features = X.shape
    X_ = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(features+1, 1)

    for epoch in range(n_epochs):
        for i in range(m):
            # Seleccionamos una instancia al azar
            random_index = np.random.randint(m)
            xi = X_[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            gradient = 2*xi.T @ (xi @ theta - yi)
            eta = learning_schedule(epoch * m + i, t0, t1)
            theta = theta - eta*gradient

    return theta