import numpy as np
from abc import ABC, abstractmethod
import funciones

'''------------------------------------MODELO GENERICO------------------------------------'''

class Genetic_Model(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
    
    @abstractmethod
    def redRate(self, X_test, y_test):
        pass
    
    @abstractmethod
    def clasRate(self):
        pass
    
    @abstractmethod
    def _most_common(self, arr: np.array):
        pass

    @abstractmethod
    def __name__(self):
        pass

'''------------------------------------MODELOS------------------------------------'''


class KNN(Genetic_Model):
    def __init__(self, k=1):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        # Calcular distancias
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Obtener los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = self.y_train[k_indices]
        # Devolver la etiqueta más común
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def _most_common(self, arr: np.array):
        return max(set(arr), key=list(arr).count)
    
    def redRate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 100*np.mean(y_pred == y_test)
    
    def clasRate(self):
        y_pred = np.array([self._predict_without_x(x) for x in self.X_train])
        return 100*np.mean(y_pred == self.y_train)
    
    def _predict_without_x(self, x):
        # Calcular distancias
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Obtener los k vecinos más cercanos
        k_indices = np.argsort(distances)[1:self.k+1]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = self.y_train[k_indices]
        # Devolver la etiqueta más común
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def __name__(self):
        return self.k+'-NN'
    
'''------------------------------------RELIEF(GREEDY)------------------------------------'''

class Relief(Genetic_Model):
    def __init__(self, k=1):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.features = np.empty(X_train.shape[1], dtype=int)
        self.weights = np.zeros(X_train.shape[1])
        self._fit()

    def _fit(self):
        for i in range(len(self.X_train)):
            distances = np.sqrt(np.sum((self.X_train - self.X_train[i]) ** 2, axis=1))
            indices = np.argsort(distances)
            nearest_hit = np.array([j for j in indices if self.y_train[i] == self.y_train[j]])[:1]
            nearest_miss = np.array([j for j in indices if self.y_train[i] != self.y_train[j]])[:1]
            for j in range(len(self.X_train[i])):
                near_hit = self.X_train[nearest_hit[0]][j]
                near_miss = self.X_train[nearest_miss[0]][j]
                self.weights[j] += abs(self.X_train[i][j] - near_miss) - abs(self.X_train[i][j] - near_hit)
        
        max_weight = max(self.weights)
        for i in range(len(self.weights)):
            if self.weights[i] < 0:
                self.weights[i] = 0
            else:
                self.weights[i] = self.weights[i] / max_weight
        self.features = np.argsort(self.weights)[::-1]

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        # Calcular distancias
        distances = np.sqrt(np.sum(((self.X_train - x) ** 2) * self.weights, axis=1))
        # Obtener los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = self.y_train[k_indices]
        # Devolver la etiqueta más común
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def _most_common(self, arr: np.array):
        return max(set(arr), key=list(arr).count)
    
    def redRate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 100*np.mean(y_pred == y_test)
    
    def clasRate(self):
        y_pred = np.array([self._predict_without_x(x) for x in self.X_train])
        return 100*np.mean(y_pred == self.y_train)
    
    def _predict_without_x(self, x):
        # Calcular distancias
        distances = np.sqrt(np.sum(((self.X_train - x) ** 2) * self.weights, axis=1))
        # Obtener los k vecinos más cercanos
        k_indices = np.argsort(distances)[1:self.k+1]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = self.y_train[k_indices]
        # Devolver la etiqueta más común
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def __name__(self):
        return 'Relief'
    

'''------------------------------------MODELO BL------------------------------------'''
    
class BL(Genetic_Model):
    def __init__(self, seed=7):
        self.seed = seed
        np.random.seed(seed)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.k = 1
        self.weights = np.random.uniform(0, 1, X_train.shape[1])
        self.features = np.empty(X_train.shape[1], dtype=int)
        self.MAX_ITER : int = 20*self.weights.shape[0]
        self._fit()

    def _fit(self):
        old_weights = np.copy(self.weights)
        actual_evaluation = self.clasRate()
        iterations : int = 0
        improves : bool = False

        while iterations < self.MAX_ITER and not improves:
            mutation_order = np.random.permutation(self.weights.shape[0])
            neighbors = self._get_neighbors(mutation_order)
            for neighbor in neighbors:
                self.weights = neighbor
                new_evaluation = self.clasRate()
                if new_evaluation > actual_evaluation:
                    actual_evaluation = new_evaluation
                    improves = True
                    break
                iterations += 1

        if not improves:
            self.weights = old_weights

        self.features = np.argsort(self.weights)[::-1]
            

    def _get_neighbors(self, mutation_order):
        random_array = np.random.normal(0, 0.2, self.weights.shape[0])
        z = self.weights + random_array
        neighbors = np.empty((mutation_order.shape[0], self.weights.shape[0]))
        for i in range(len(mutation_order)):
            neighbor = np.copy(self.weights)
            neighbor[mutation_order[i]] = z[i]

            if neighbor[mutation_order[i]] < 0:
                neighbor[mutation_order[i]] = 0
            elif neighbor[mutation_order[i]] > 1:
                neighbor[mutation_order[i]] = 1

            neighbors[i] = neighbor

        return neighbors
    
    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])
    
    def _predict(self, x):
        # Calcular distancias
        distances = np.sqrt(np.sum(((self.X_train - x) ** 2) * self.weights, axis=1))
        # Obtener los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = self.y_train[k_indices]
        # Devolver la etiqueta más común
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
    def _most_common(self, arr: np.array):
        return max(set(arr), key=list(arr).count)

    def redRate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 100*np.mean(y_pred == y_test)
    
    def clasRate(self):
        y_pred = np.array([self._predict_without_x(x) for x in self.X_train])
        return 100*np.mean(y_pred == self.y_train)
    
    def _predict_without_x(self, x):
        # Calcular distancias
        distances = np.sqrt(np.sum(((self.X_train - x) ** 2) * self.weights, axis=1))
        # Obtener los k vecinos más cercanos
        k_indices = np.argsort(distances)[1:self.k+1]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = self.y_train[k_indices]
        # Devolver la etiqueta más común
        most_common = self._most_common(k_nearest_labels)
        return most_common
    
        