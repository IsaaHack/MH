import subprocess

try:
    import numpy as np
except ImportError:
    print("NumPy no está instalado. Instalando NumPy...")
    try:
        subprocess.check_call(["pip", "install", "numpy"])
        import numpy as np
        print("NumPy se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar NumPy. Por favor, instálalo manualmente.")



from abc import ABC, abstractmethod

try:
    import scipy.spatial.distance as sp
except ImportError:
    print("SciPy no está instalado. Instalando SciPy...")
    try:
        subprocess.check_call(["pip", "install", "scipy"])
        import scipy.spatial.distance as sp
        print("SciPy se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar SciPy. Por favor, instálalo manualmente.")

SQRT_03 = np.sqrt(0.3)

'''------------------------------------MODELO GENERICO------------------------------------'''

class Genetic_Model():
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
        w = np.copy(self.weights)
        w[w < 0.1] = 0

        X = np.concatenate((self.X_train, X_test))

        distances = sp.squareform(sp.pdist(X, 'euclidean', w=w))
        distances[np.diag_indices(distances.shape[0])] = np.inf
        distances = distances[self.X_train.shape[0]:, :self.X_train.shape[0]]
        index_predictions = np.argmin(distances, axis=1)
        predictions_labels = self.y_train[index_predictions]

        return predictions_labels
    
    @abstractmethod
    def _red_rate(self, weights):
        return 100*np.sum(weights < 0.1)/weights.shape[0]
    
    @abstractmethod
    def red_rate(self):
        return self._red_rate(self.weights)
    
    @abstractmethod
    def _clas_rate(self, weights):
        w = np.copy(weights)
        w[w < 0.1] = 0
        distances = sp.squareform(sp.pdist(self.X_train, 'euclidean', w=w))
        distances[np.diag_indices(distances.shape[0])] = np.inf
        index_predictions = np.argmin(distances, axis=1)
        predictions_labels = self.y_train[index_predictions]

        return 100*np.mean(predictions_labels == self.y_train)
    
    @abstractmethod
    def clas_rate(self):
        return self._clas_rate(self.weights)
    
    @abstractmethod
    def _fitness(self, weights, alpha=0.75):
        return self._clas_rate(weights) * alpha + self._red_rate(weights) * (1-alpha)
    
    
    @abstractmethod
    def fitness(self, clasRate, redRate, alpha=0.75):
        return clasRate * alpha + redRate * (1-alpha)
    
    @abstractmethod
    def global_score(self, alpha=0.75):
        clasRate = self.clas_rate(self.weights)
        redRate = self.red_rate(self.weights)
        return clasRate, redRate, self.fitness(clasRate, redRate, alpha)
    
    @abstractmethod
    def accuracy(self, X_test, y_test):
        return 100*np.mean(self.predict(X_test) == y_test)

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
        X = np.concatenate((self.X_train, X_test))

        distances = sp.squareform(sp.pdist(X, 'euclidean'))
        distances[np.diag_indices(distances.shape[0])] = np.inf
        distances = distances[self.X_train.shape[0]:, :self.X_train.shape[0]]

        if self.k == 1:
            k_indices = np.argmin(distances, axis=1)
            return self.y_train[k_indices]
        
        k_indices = np.argsort(distances, axis=1)
        k_indices = k_indices[:,:self.k]
        k_nearest_labels = self.y_train[k_indices]

        return np.array([self._most_common(k_nearest_labels[i]) for i in range(k_nearest_labels.shape[0])])
    
    def _most_common(self, arr: np.array):
        return max(set(arr), key=list(arr).count)
    
    def _red_rate(self, weights):
        return super()._red_rate(weights)
    
    def red_rate(self):
        return super().red_rate()
    
    def _clas_rate(self, weights):
        distances = sp.squareform(sp.pdist(self.X_train, 'euclidean'))
        distances[np.diag_indices(distances.shape[0])] = np.inf
        
        if self.k == 1:
            k_indices = np.argmin(distances, axis=1)
            predictions =  self.y_train[k_indices]
            return 100*np.mean(predictions == self.y_train)
        
        k_indices = np.argsort(distances, axis=1)
        k_indices = k_indices[:,:self.k]
        k_nearest_labels = self.y_train[k_indices]
        predictions = np.array([self._most_common(k_nearest_labels[i]) for i in range(k_nearest_labels.shape[0])])

        return 100*np.mean(predictions == self.y_train)
    
    def clas_rate(self):
        return self._clas_rate(self.weights)
    
    def _fitness(self, weights, alpha=0.75):
        return super()._fitness(weights, alpha)
    
    def fitness(self, clasRate, redRate, alpha=0.75):
        return super().fitness(clasRate, redRate, alpha)
    
    def global_score(self, alpha=0.75):
       return super().global_score(alpha)
    
    def accuracy(self, X_test, y_test):
        return 100*np.mean(self.predict(X_test) == y_test)
    
    def __name__(self):
        return self.k+'-NN'
    
'''------------------------------------RELIEF(GREEDY)------------------------------------'''

class Relief(Genetic_Model):
    def __init__(self):
        super().__init__()
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.zeros(X_train.shape[1])
        self._fit()

    def _fit(self):
        distances = sp.squareform(sp.pdist(self.X_train, 'euclidean'))
        #distances[np.diag_indices(distances.shape[0])] = np.inf
        distances[distances == 0] = np.inf
        all_index = np.argsort(distances, axis=1)

        for i in range(len(self.X_train)):
            n_hits = all_index[i][self.y_train[i] == self.y_train[all_index[i]]]
            #n_hits = n_hits[distances[i][n_hits] != 0]
            try:
                nearest_hit = n_hits[0]
            except:
                nearest_hit = -1
            nearest_miss = all_index[i][self.y_train[i] != self.y_train[all_index[i]]][0]

            if nearest_hit != -1 and distances[i][nearest_hit] != np.inf:
                self.weights += np.abs(self.X_train[i] - self.X_train[nearest_miss]) - np.abs(self.X_train[i] - self.X_train[nearest_hit])
        
        max_weight = np.max(self.weights)
        self.weights = np.maximum(self.weights / max_weight, 0)

    def predict(self, X_test):
        return super().predict(X_test)
    
    def _red_rate(self, weights):
        return super()._red_rate(weights)
    
    def red_rate(self):
        return super().red_rate()
    
    def _clas_rate(self, weights):
        return super()._clas_rate(weights)
    
    def clas_rate(self):
        return super().clas_rate()
    
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
        self.weights[self.weights < 0.1] = 0
        self.MAX_ITER : int = 20*self.weights.shape[0]
        self._fit(evaluations)

    def _fit(self, evaluations=15000):
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

                if new_evaluation > actual_evaluation:
                    actual_evaluation = new_evaluation
                    self.weights = np.copy(neighbor)
                    iterations = 0
            

    def _get_neighbor(self, mutation_order):
        mutation = np.random.normal(0, SQRT_03)
        neighbor = np.copy(self.weights)

        neighbor[mutation_order] = np.clip(neighbor[mutation_order] + mutation, 0, 1)

        return neighbor
    
    def predict(self, X_test):
        return super().predict(X_test)
    
    def _red_rate(self, weights):
        return super()._red_rate(weights)
    
    def red_rate(self):
        return super().red_rate()
    
    def _clas_rate(self, weights):
        return super()._clas_rate(weights)
    
    def clas_rate(self):
        return super().clas_rate()
    
    def fitness(self, clasRate, redRate, alpha=0.75):
        return super().fitness(clasRate, redRate, alpha)
    
    def _fitness(self, weights, alpha=0.75):
        return super()._fitness(weights, alpha)
    
    def global_score(self, alpha=0.75):
        return super().global_score(alpha)
    
    def accuracy(self, X_test, y_test):
        return super().accuracy(X_test, y_test)
    
    
    
    
        