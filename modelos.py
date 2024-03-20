import numpy as np
from abc import ABC, abstractmethod
import funciones

SQRT_03 = np.sqrt(0.3)

'''------------------------------------MODELO GENERICO------------------------------------'''

class Genetic_Model(ABC):
    @abstractmethod
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.weights = None

    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        y_pred = np.array([self._predict(x, self.weights) for x in X_test])
        return y_pred

    @abstractmethod
    def _predict(self, x, weights):
        distances = np.sqrt(np.sum((((self.X_train - x)) ** 2)* weights , axis=1))
        return self.y_train[np.argmin(distances)]

    @abstractmethod
    def _predict_without_x(self, x, weights):
        distances = np.sqrt(np.sum((((self.X_train - x)) ** 2)* weights , axis=1))
        nearest = np.argsort(distances)[1:2]
        return self.y_train[nearest[0]]
    
    @abstractmethod
    def _redRate(self, weights):
        contador = 0
        contador += np.sum(weights < 0.1)
        return 100*contador/weights.shape[0]
    
    @abstractmethod
    def redRate(self):
        return self._redRate(self.weights)
    
    @abstractmethod
    def _clasRate(self, weights):
        y_pred = np.apply_along_axis(self._predict_without_x, 1, self.X_train, weights)
        return 100*np.mean(y_pred == self.y_train)
    
    @abstractmethod
    def clasRate(self):
        return self._clasRate(self.weights)
    
    @abstractmethod
    def _fitness(self, weights, alpha=0.75):
        return self._clasRate(weights) * alpha + self._redRate(weights) * (1-alpha)
    
    @abstractmethod
    def fitness(self, clasRate, redRate, alpha=0.75):
        return clasRate * alpha + redRate * (1-alpha)
    
    @abstractmethod
    def global_score(self, alpha=0.75):
        clasRate = self.clasRate(self.weights)
        redRate = self.redRate(self.weights)
        return clasRate, redRate, self.fitness(clasRate, redRate, alpha)
    
    @abstractmethod
    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 100*np.mean(y_pred == y_test)

    @abstractmethod
    def __name__(self):
        pass

'''------------------------------------MODELOS------------------------------------'''


class KNN(Genetic_Model):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.ones(X_train.shape[1])

    def predict(self, X_test):
        return np.apply_along_axis(self._predict, 1, X_test, self.weights)

    def _predict(self, x, weights):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        if self.k == 1:
            return self.y_train[k_indices[0]]
        k_nearest_labels = self.y_train[k_indices]
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def _predict_without_x(self, x, weights):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[1:self.k+1]
        if self.k == 1:
            return self.y_train[k_indices[0]]
        k_nearest_labels = self.y_train[k_indices]
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def _most_common(self, arr: np.array):
        return max(set(arr), key=list(arr).count)
    
    def _redRate(self, weights):
        return super()._redRate(weights)
    
    def redRate(self):
        return super().redRate()
    
    def _clasRate(self, weights):
        return super()._clasRate(weights)
    
    def clasRate(self):
        return super().clasRate()
    
    def _fitness(self, weights, alpha=1):
        return super()._fitness(weights, alpha)
    
    def fitness(self, clasRate, redRate, alpha=1):
        return super().fitness(clasRate, redRate, alpha)
    
    def global_score(self, alpha=1):
        return super().global_score(alpha)
    
    def accuracy(self, X_test, y_test):
        return super().accuracy(X_test, y_test)
    
    def __name__(self):
        return self.k+'-NN'
    
'''------------------------------------RELIEF(GREEDY)------------------------------------'''

class Relief(Genetic_Model):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.features = np.empty(X_train.shape[1], dtype=int)
        self.weights = np.zeros(X_train.shape[1])
        self.true_weights = np.zeros(X_train.shape[1])
        self._fit()

    def _fit(self):
        for i in range(len(self.X_train)):
            distances = np.sqrt(np.sum((self.X_train - self.X_train[i]) ** 2, axis=1))
            indices = np.argsort(distances)[1:]
            nearest_hit = -1

            n_hits = indices[self.y_train[i] == self.y_train[indices]]
            if n_hits.size > 0:
                nearest_hit = n_hits[0]
            nearest_miss = indices[self.y_train[i] != self.y_train[indices]][0]

            if nearest_hit != -1:
                self.weights += np.abs(self.X_train[i] - self.X_train[nearest_miss]) - np.abs(self.X_train[i] - self.X_train[nearest_hit])
        
        max_weight = np.max(self.weights)
        self.weights = np.maximum(self.weights / max_weight, 0) 
        self.true_weights = np.copy(self.weights)
        self.features = np.argsort(self.true_weights)[::-1]
        self.weights[self.weights < 0.1] = 0

    def predict(self, X_test):
        return super().predict(X_test)

    def _predict(self, x, weights):
        return super()._predict(x, weights)
    
    def _predict_without_x(self, x, weights):
        return super()._predict_without_x(x, weights)
    
    def _redRate(self, weights):
        return super()._redRate(weights)
    
    def redRate(self):
        return super().redRate()
    
    def _clasRate(self, weights):
        return super()._clasRate(weights)
    
    def clasRate(self):
        return super().clasRate()
    
    def _fitness(self, weights, alpha=0.75):
        return super()._fitness(weights, alpha)
    
    def fitness(self, clasRate, redRate, alpha=0.75):
        return super().fitness(clasRate, redRate, alpha)
    
    def global_score(self, alpha=0.75):
        return super().global_score(alpha)
    
    def accuracy(self, X_test, y_test):
        return super().accuracy(X_test, y_test)
    
    def __name__(self):
        return 'Relief'
    

'''------------------------------------MODELO BL------------------------------------'''
    
class BL(Genetic_Model):
    def __init__(self, seed=7):
        super().__init__()
        self.seed = seed
        np.random.seed(seed)

    def fit(self, X_train, y_train, evaluations=15000):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.uniform(0, 1, X_train.shape[1])
        self.features = np.empty(X_train.shape[1], dtype=int)
        self.MAX_ITER : int = 20*self.weights.shape[0]
        self._fit(evaluations)

    def _fit(self, evaluations=15000):
        old_weights = np.copy(self.weights)
        actual_evaluation = self._fitness(self.weights)
        iterations : int = 0
        n_eval : int = 0

        while iterations < self.MAX_ITER and n_eval < evaluations:
            mutation_order = np.random.permutation(self.weights.shape[0])

            for mut in mutation_order:
                neighbor = self._get_neighbor(mut)
                new_evaluation = self._fitness(neighbor)
                iterations += 1
                n_eval += 1
                while new_evaluation > actual_evaluation and n_eval < evaluations:
                    actual_evaluation = new_evaluation
                    self.weights = np.copy(neighbor)
                    iterations = 0
                    neighbor = self._get_neighbor(mut)
                    new_evaluation = self._fitness(neighbor)
                    n_eval += 1

                if n_eval >= evaluations or iterations >= self.MAX_ITER:
                    break
                

        self.features = np.argsort(self.weights)[::-1]
            

    def _get_neighbor(self, mutation_order):
        mutation = np.random.normal(0, SQRT_03)
        
        neighbor = np.copy(self.weights)

        neighbor[mutation_order] += mutation

        neighbor[neighbor < 0.1] = 0

        neighbor[neighbor > 1] = 1

        return neighbor
    
    def predict(self, X_test):
        return super().predict(X_test)

    def _predict(self, x, weights):
        return super()._predict(x, weights)
    
    def _predict_without_x(self, x, weights):
        return super()._predict_without_x(x, weights)
    
    def _redRate(self, weights):
        return super()._redRate(weights)
    
    def redRate(self):
        return super().redRate()
    
    def _clasRate(self, weights):
        return super()._clasRate(weights)
    
    def clasRate(self):
        return super().clasRate()
    
    def fitness(self, clasRate, redRate, alpha=0.75):
        return super().fitness(clasRate, redRate, alpha)
    
    def _fitness(self, weights, alpha=0.75):
        return super()._fitness(weights, alpha)
    
    def global_score(self, alpha=0.75):
        return super().global_score(alpha)
    
    def accuracy(self, X_test, y_test):
        return super().accuracy(X_test, y_test)
    
    
    
        