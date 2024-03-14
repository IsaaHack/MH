import numpy as np

ALPHA = 0.75

def evaluationFunction(tasa_clas, tasa_red, alfa=0.75):
    return alfa*tasa_clas + (1-alfa)*tasa_red

def getEuclideanDistances(X, x):
    return np.sqrt(np.sum((X - x) ** 2, axis=1))

def getWeightedEuclideanDistances(X : np.array, W : np.array, x):
    return np.sqrt(np.sum(W * (X - x) ** 2, axis=1))
