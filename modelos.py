import subprocess
from abc import ABC, abstractmethod

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

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm no está instalado. Instalando tqdm...")
    try:
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm
        print("tqdm se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar tqdm. Por favor, instálalo manualmente.")

SQRT_03 : float = np.sqrt(0.3)
DEFAULT_MAX_EVAL : int = 15000

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

    def predict(self, X_test : np.ndarray):
        X = np.concatenate((self.X_train, X_test))

        weights_to_use = self.weights[self.weights >= 0.1]
        atributes_to_use = X[:, self.weights >= 0.1]

        distances = sp.squareform(sp.pdist(atributes_to_use, 'euclidean', w=weights_to_use))
        distances[np.diag_indices(distances.shape[0])] = np.inf
        distances = distances[self.X_train.shape[0]:, :self.X_train.shape[0]]
        index_predictions = np.argmin(distances, axis=1)
        predictions_labels = self.y_train[index_predictions]

        return predictions_labels

    def _red_rate(self, weights : np.ndarray):
        return 100*np.sum(weights < 0.1)/weights.shape[0]

    def red_rate(self):
        return self._red_rate(self.weights)

    def _clas_rate(self, weights : np.ndarray):
        weights_to_use = weights[weights >= 0.1]
        atributes_to_use = self.X_train[:, weights >= 0.1]

        distances = sp.squareform(sp.pdist(atributes_to_use, 'euclidean', w=weights_to_use))
        distances[np.diag_indices(distances.shape[0])] = np.inf
        index_predictions = np.argmin(distances, axis=1)
        predictions_labels = self.y_train[index_predictions]

        return 100*np.mean(predictions_labels == self.y_train)

    def clas_rate(self):
        return self._clas_rate(self.weights)

    def _fitness(self, weights : np.ndarray, alpha : float = 0.75):
        return self._clas_rate(weights) * alpha + self._red_rate(weights) * (1-alpha)

    def fitness(self, clasRate : float, redRate : float, alpha : float = 0.75):
        return clasRate * alpha + redRate * (1-alpha)

    def global_score(self, alpha : float = 0.75):
        clasRate = self.clas_rate(self.weights)
        redRate = self.red_rate(self.weights)
        return clasRate, redRate, self.fitness(clasRate, redRate, alpha)

    def accuracy(self, X_test : np.ndarray, y_test : np.ndarray):
        return 100*np.mean(self.predict(X_test) == y_test)
    
    def _fit_Relief(self, weights : np.ndarray[float]):
        distances = sp.squareform(sp.pdist(self.X_train, 'euclidean'))
        distances[distances == 0] = np.inf
        all_index = np.argsort(distances, axis=1)

        for i in range(len(self.X_train)):
            n_hits = all_index[i][self.y_train[i] == self.y_train[all_index[i]]]
            try:
                nearest_hit = n_hits[0]
            except:
                nearest_hit = -1
            nearest_miss = all_index[i][self.y_train[i] != self.y_train[all_index[i]]][0]

            if nearest_hit != -1 and distances[i][nearest_hit] != np.inf:
                weights += np.abs(self.X_train[i] - self.X_train[nearest_miss]) - np.abs(self.X_train[i] - self.X_train[nearest_hit])

        max_weight = np.max(weights)
        weights = np.maximum(weights / max_weight, 0)

        return weights
    
    def _fit_BL(self, weights : np.ndarray, fitness = None, max_iter : int = DEFAULT_MAX_EVAL, max_evaluations : int = DEFAULT_MAX_EVAL, pb : bool = False):
        if fitness is None:
            actual_evaluation = self._fitness(weights)
        else:
            actual_evaluation = fitness
        iterations : int = 0
        n_eval : int = 0

        if pb:
            progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)
        else:
            progress_bar = None

        while iterations < max_iter and n_eval < max_evaluations:
            mutation_order = np.random.permutation(weights.shape[0])

            for mut in mutation_order:
                neighbor = self._get_neighbor(weights, mut)
                new_evaluation = self._fitness(neighbor)
                iterations += 1
                n_eval += 1

                if(pb): progress_bar.update(1)

                if new_evaluation > actual_evaluation:
                    actual_evaluation = new_evaluation
                    weights = np.copy(neighbor)
                    iterations = 0

                if n_eval >= max_evaluations:
                    break

        if(pb): progress_bar.update(max_evaluations - n_eval)

        return weights, actual_evaluation, n_eval
    
    def _get_neighbor(self, weights : np.ndarray, mutation_order : int):
        mutation = np.random.normal(0, SQRT_03)
        neighbor = np.copy(weights)

        neighbor[mutation_order] = np.clip(neighbor[mutation_order] + mutation, 0, 1)

        return neighbor


'''------------------------------------MODELOS------------------------------------'''

class KNN(Genetic_Model):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def fit(self, X_train : np.ndarray, y_train : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.weights : np.ndarray = np.ones(X_train.shape[1])

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

    def _most_common(self, arr: np.array):
        return max(set(arr), key=list(arr).count)

    def clas_rate(self):
        return self._clas_rate(self.weights)

    def accuracy(self, X_test, y_test):
        return 100*np.mean(self.predict(X_test) == y_test)


'''------------------------------------RELIEF(GREEDY)------------------------------------'''

class Relief(Genetic_Model):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.zeros(X_train.shape[1])
        self.weights = self._fit_Relief(self.weights)

'''------------------------------------MODELO BL------------------------------------'''

class BL(Genetic_Model):
    def __init__(self, seed : int = 7, evaluations : int = 15000):
        super().__init__()
        self.seed = seed
        self.MAX_EVAL = evaluations
        np.random.seed(seed)

    def fit(self, X_train : np.ndarray, y_train : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.uniform(0, 1, X_train.shape[1])
        MAX_ITER = 20*self.weights.shape[0]
        self.weights, fitness, n_eval = self._fit_BL(self.weights, max_iter=MAX_ITER, max_evaluations=self.MAX_EVAL, pb=True)

    # def _fit(self, evaluations=15000):
    #     actual_evaluation = self._fitness(self.weights)
    #     iterations : int = 0
    #     n_eval : int = 0

    #     progress_bar = tqdm(total=evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

    #     while iterations < self.MAX_ITER and n_eval < evaluations:
    #         mutation_order = np.random.permutation(self.weights.shape[0])

    #         for mut in mutation_order:
    #             neighbor = self._get_neighbor(mut)
    #             new_evaluation = self._fitness(neighbor)
    #             iterations += 1
    #             n_eval += 1
    #             progress_bar.update(1)

    #             if new_evaluation > actual_evaluation:
    #                 actual_evaluation = new_evaluation
    #                 self.weights = np.copy(neighbor)
    #                 iterations = 0

    #             if n_eval >= evaluations:
    #                 break

    #     progress_bar.update(evaluations - n_eval)


    # def _get_neighbor(self, mutation_order):
    #     mutation = np.random.normal(0, SQRT_03)
    #     neighbor = np.copy(self.weights)

    #     neighbor[mutation_order] = np.clip(neighbor[mutation_order] + mutation, 0, 1)

    #     return neighbor


'''------------------------------------MODELO AGG------------------------------------'''

class AGG(Genetic_Model):
    def __init__(self, crossover_function, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 0.7, improved : bool = False):
        super().__init__()
        self.seed = seed
        self.crossover_function = crossover_function
        self.population = population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.MAX_EVAL = evaluations
        self.improved = improved
        np.random.seed(seed)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.empty(X_train.shape[1])
        self._fit(self.crossover_function, self.MAX_EVAL, self.improved)

    def _fit(self, crossover_function, max_evaluations=15000, improved=False):
        eval : int = 0
        progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

        if improved:
            selection_funtion = self._best_selection
        else:
            selection_funtion = self._selection

        population = np.random.uniform(0, 1, (self.population, self.weights.shape[0]))
        fitnesess = np.apply_along_axis(self._fitness, 1, population)
        eval += self.population
        progress_bar.update(self.population)
        best = population[np.argmax(fitnesess)].copy()
        best_fit = fitnesess[np.argmax(fitnesess)]

        while eval < max_evaluations:
            new_population = selection_funtion(population, fitnesess)
            crossover_function(new_population, self.crossover_rate)
            self._mutation(new_population, self.mutation_rate)

            fitnesess = np.apply_along_axis(self._fitness, 1, new_population)
            eval += self.population
            progress_bar.update(self.population)

            best_new = new_population[np.argmax(fitnesess)]
            best_new_fit = fitnesess[np.argmax(fitnesess)]

            if best_new_fit > best_fit:
                best = best_new
                best_fit = best_new_fit
            else:
                new_population[np.argmin(fitnesess)] = best
                fitnesess[np.argmin(fitnesess)] = best_fit

            population = new_population

        self.weights = np.copy(best)

    def _selection(self, population, fitnesess):
        new_population = np.empty((self.population, self.weights.shape[0]))

        random_indexes = np.empty((self.population, 3), dtype=int)

        for i in range(random_indexes.shape[0]):
            random_indexes[i] = np.random.choice(self.population, 3, replace=False)

        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(self.population), best_in_each_group]]

        return new_population

    def _best_selection(self, population, fitnesess):
        new_population = np.empty((self.population, self.weights.shape[0]))

        random_indexes = np.random.randint(0, self.population, size=(self.population, 3))
        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(self.population), best_in_each_group]]

        return new_population

    def _mutation(self, population, mutation_rate):
        estimated_mutations = int(mutation_rate * population.size)

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)


'''------------------------------------MODELO AGE------------------------------------'''

class AGE(Genetic_Model):
    def __init__(self, crossover_function, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 1, improved : bool = False):
        super().__init__()
        self.seed = seed
        self.crossover_function = crossover_function
        self.population = population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.MAX_EVAL = evaluations
        self.improved = improved
        np.random.seed(seed)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.empty(X_train.shape[1])
        self._fit(self.crossover_function, self.MAX_EVAL, self.improved)

    def _fit(self, crossover_function, max_evaluations=15000, improved=False):
        eval : int = 0
        progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

        if improved:
            selection_funtion = self._best_selection
        else:
            selection_funtion = self._selection

        population = np.random.uniform(0, 1, (self.population, self.weights.shape[0]))
        fitnesess = np.apply_along_axis(self._fitness, 1, population)
        eval += self.population
        progress_bar.update(self.population)

        while eval < max_evaluations:
            childrens = selection_funtion(population, fitnesess)
            crossover_function(childrens, self.crossover_rate)
            self._mutation(childrens, self.mutation_rate)

            fitnesess_children = np.apply_along_axis(self._fitness, 1, childrens)
            eval += childrens.shape[0]
            progress_bar.update(childrens.shape[0])

            big_brother = np.argmax(fitnesess_children)
            little_brother = np.argmin(fitnesess_children)

            worst = np.argmin(fitnesess)
            if(fitnesess_children[big_brother] > fitnesess[worst]):
                population[worst] = childrens[big_brother]
                fitnesess[worst] = fitnesess_children[big_brother]

                worst = np.argmin(fitnesess)
                if(fitnesess_children[little_brother] > fitnesess[worst]):
                    population[worst] = childrens[little_brother]
                    fitnesess[worst] = fitnesess_children[little_brother]

        self.weights =  np.copy(population[np.argmax(fitnesess)])


    def _best_selection(self, population, fitnesess):
        new_population = np.empty((2, self.weights.shape[0]))

        random_indexes = np.random.choice(self.population, (2), replace=False)

        new_population = population[random_indexes]

        return new_population
    
    def _selection(self, population, fitnesess):
        new_population = np.empty((2, self.weights.shape[0]))

        random_indexes = np.empty((2, 3), dtype=int)

        for i in range(random_indexes.shape[0]):
            random_indexes[i] = np.random.choice(self.population, 3, replace=False)

        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)

        new_population = np.copy(population[random_indexes[np.arange(2), best_in_each_group]])

        return new_population

    def _mutation(self, population, mutation_rate):
        random_chances = np.random.uniform(0, 1, population.shape[0])

        population[random_chances < mutation_rate] = np.clip(np.random.normal(0, SQRT_03) + population[random_chances < mutation_rate], 0, 1)


'''------------------------------------MODELO AM------------------------------------'''

class AM(Genetic_Model):
    def __init__(self, crossover_function, bl_selection_function, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 0.7, improved : bool = False):
        super().__init__()
        self.seed = seed
        self.crossover_function = crossover_function
        self.bl_selection_function = bl_selection_function
        self.MAX_EVAL = evaluations
        self.population = population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.improved = improved
        np.random.seed(seed)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.empty(X_train.shape[1])
        self._fit(self.crossover_function, self.bl_selection_function, max_evaluations=self.MAX_EVAL, improved=self.improved)

    def _fit(self, crossover_funtion, bl_selection_function, bl_rate : int = 10, max_evaluations=15000, improved=False):
        eval : int = 0
        n_generations : int = 1
        bl_evaluations : int = self.weights.shape[0] * 2
        bl_iterations : int = bl_evaluations

        if improved:
            selection_funtion = self._best_selection
        else:
            selection_funtion = self._selection

        progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

        population = np.random.uniform(0, 1, (self.population, self.weights.shape[0]))
        fitnesess = np.apply_along_axis(self._fitness, 1, population)
        eval += self.population
        progress_bar.update(self.population)
        best = population[np.argmax(fitnesess)].copy()
        best_fit = fitnesess[np.argmax(fitnesess)]

        while eval < max_evaluations:
            new_population = selection_funtion(population, fitnesess)
            crossover_funtion(new_population, self.crossover_rate)
            self._mutation(new_population, self.mutation_rate)

            fitnesess = np.apply_along_axis(self._fitness, 1, new_population)
            eval += self.population
            progress_bar.update(self.population)

            if n_generations % bl_rate == 0:
                bl_selection = bl_selection_function(new_population, fitnesess)

                if max_evaluations - eval >= bl_evaluations * bl_selection.size:
                    #bl_evaluations = (max_evaluations - eval) // bl_selection.size
                
                    for index in bl_selection:
                        new_population[index], fitnesess[index], n_eval_bl = self._fit_BL(weights=new_population[index], fitness=fitnesess[index], max_iter=bl_iterations, max_evaluations=bl_evaluations, pb=False)
                        
                        eval += n_eval_bl
                        progress_bar.update(n_eval_bl)


            best_new = new_population[np.argmax(fitnesess)]
            best_new_fit = fitnesess[np.argmax(fitnesess)]

            if best_new_fit > best_fit:
                best = best_new
                best_fit = best_new_fit
            else:
                new_population[np.argmin(fitnesess)] = best
                fitnesess[np.argmin(fitnesess)] = best_fit

            population = new_population
            n_generations += 1

        self.weights = np.copy(best)


    def _best_selection(self, population, fitnesess):
        new_population = np.empty((self.population, self.weights.shape[0]))

        random_indexes = np.random.randint(0, self.population, size=(self.population, 3))
        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(self.population), best_in_each_group]]

        return new_population
    
    def _selection(self, population, fitnesess):
        new_population = np.empty((self.population, self.weights.shape[0]))

        random_indexes = np.empty((self.population, 3), dtype=int)

        for i in range(random_indexes.shape[0]):
            random_indexes[i] = np.random.choice(self.population, 3, replace=False)

        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(self.population), best_in_each_group]]

        return new_population

    def _mutation(self, population, mutation_rate):
        estimated_mutations = int(mutation_rate * population.size)

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)