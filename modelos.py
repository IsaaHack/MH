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

import multiprocessing as mp
import psutil

SQRT_03 : float = np.sqrt(0.3)
DEFAULT_MAX_EVAL : int = 15000

'''------------------------------------MODELO GENERICO------------------------------------'''

class Generic_Model(ABC):
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

        #atributes_to_use = atributes_to_use * np.sqrt(weights_to_use)

        # equivalente a la distancia euclidea
        distances = sp.squareform(sp.pdist(atributes_to_use, 'minkowski', p=2, w=weights_to_use))
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

        #atributes_to_use = atributes_to_use * np.sqrt(weights_to_use)
        
        # equivalente a la distancia euclidea
        distances = sp.squareform(sp.pdist(atributes_to_use, 'minkowski', p=2, w=weights_to_use))
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
    
    
    def _fit_BL(self, weights : np.ndarray, fitness = None, max_iter : int = DEFAULT_MAX_EVAL, max_evaluations : int = DEFAULT_MAX_EVAL, pb : bool = False):
        iterations : int = 0
        n_eval : int = 0

        if pb: progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)
        else: progress_bar = None
        
        if fitness is None: 
            actual_evaluation = self._fitness(weights)
            n_eval += 1
            if pb: progress_bar.update(1)
        else: 
            actual_evaluation = fitness

        while iterations < max_iter and n_eval < max_evaluations:
            mutation_order = np.random.permutation(weights.shape[0])

            for mut in mutation_order:
                neighbor = self._get_neighbor(weights, mut)
                new_evaluation = self._fitness(neighbor)
                iterations += 1
                n_eval += 1

                if pb: progress_bar.update(1)

                if new_evaluation > actual_evaluation:
                    actual_evaluation = new_evaluation
                    weights = np.copy(neighbor)
                    iterations = 0

                if n_eval >= max_evaluations or iterations >= max_iter:
                    break

        if pb: progress_bar.update(max_evaluations - n_eval)

        return weights, actual_evaluation, n_eval
    
    def _get_neighbor(self, weights : np.ndarray, mutation_order : int):
        mutation = np.random.normal(0, SQRT_03)
        neighbor = np.copy(weights)

        neighbor[mutation_order] = np.clip(neighbor[mutation_order] + mutation, 0, 1)

        return neighbor
    
    def _fit_ES(self, weights : np.ndarray, fitness = None, max_iter : int = DEFAULT_MAX_EVAL, max_evaluations : int = DEFAULT_MAX_EVAL, pb : bool = False):
        n_eval : int = 0

        if pb: progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)
        else: progress_bar = None

        if fitness is None:
            actual_evaluation = self._fitness(weights)
            n_eval += 1
            if pb: progress_bar.update(1)
        else:
            actual_evaluation = fitness

        actual_weights = np.copy(weights)   

        best = np.copy(weights)
        best_evaluation = actual_evaluation

        temperature : float = self._inicial_temperature(sigma=0.3, mu=0.1, eval=actual_evaluation)
        final_temperature : float = 10e-3

        max_neighbours : int = 10*weights.shape[0]
        max_successes : int = max_neighbours // 10
        max_coolings : int = max_evaluations // max_neighbours

        beta : float = (temperature - final_temperature)/(max_coolings*temperature*final_temperature)

        improvement : bool = True

        while improvement and n_eval < max_evaluations:
            n_neighbours : int = 0
            n_successes : int = 0

            while n_neighbours < max_neighbours and n_successes < max_successes and n_eval < max_evaluations:
                mut = np.random.randint(0, weights.shape[0])

                neighbor = self._get_neighbor(actual_weights, mut)
                new_evaluation = self._fitness(neighbor)
                n_eval += 1
                n_neighbours += 1

                if(pb): progress_bar.update(1)

                delta = actual_evaluation - new_evaluation

                if delta < 0 or np.random.uniform(0, 1) <= np.exp(-delta/temperature):
                    actual_weights = np.copy(neighbor)
                    actual_evaluation = new_evaluation

                    n_successes += 1
                    
                    if actual_evaluation > best_evaluation:
                        best = np.copy(actual_weights)
                        best_evaluation = actual_evaluation

            temperature = self._cooling(temperature, beta)
            improvement = n_successes > 0

        if(pb): progress_bar.update(max_evaluations - n_eval)

        return best, best_evaluation, n_eval
    
    def _cooling(self, temperature : float, beta : float):
        return temperature/(1 + beta*temperature)

    def _inicial_temperature(self, sigma : float, mu : float, eval : float):
        return mu*eval/-np.log(sigma)
 

'''------------------------------------MODELOS------------------------------------'''

class KNN(Generic_Model):
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

class Relief(Generic_Model):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.zeros(X_train.shape[1])
        self.weights = self._fit_Relief(self.weights)

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

'''------------------------------------MODELO BL------------------------------------'''

class BL(Generic_Model):
    def __init__(self, seed : int = 7, evaluations : int = 15000):
        super().__init__()
        self.seed = seed
        self.params = {
            'max_evaluations' : evaluations,
            'pb' : True
        }
        np.random.seed(seed)

    def fit(self, X_train : np.ndarray, y_train : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.uniform(0, 1, X_train.shape[1])

        self.params['weights'] = self.weights
        self.params['max_iter'] = 20*self.weights.shape[0]

        self.weights, fitness, n_eval = self._fit_BL(**self.params)


'''------------------------------------MODELO GENETICO------------------------------------'''

class Genetic(Generic_Model):
    @abstractmethod
    def __init__(self, crossover : str, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 0.7):
        super().__init__()
        self.seed = seed
        self._crossover = self._get_crossover_function(crossover)
        self.params = {
            'max_evaluations' : evaluations,
            'n_people' : population,
            'crossover_rate' : crossover_rate,
            'mutation_rate' : mutation_rate
        }
        np.random.seed(seed)

    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    
    @abstractmethod
    def _selection(self, population, fitnesess):
        pass

    def _crossover(self, population, crossover_rate):
        pass
    
    @abstractmethod
    def _mutation(self, population, mutation_rate):
        pass

    def _get_crossover_function(self, croosover : str):
        croosover = croosover.upper()

        match croosover:
            case 'CA':
                return self._crossoverCA
            case 'BLX':
                return self._crossoverBLX
            case _:
                exception = ValueError(f"El crossover {croosover} no está implementado.")
                raise exception

    def _crossoverCA(self, population, crossover_rate):
        estimated_crossovers : int = np.floor(crossover_rate * len(population) / 2).astype(int)
        alphas = np.random.uniform(0, 1, estimated_crossovers*2)

        for i in range(estimated_crossovers):
            parent1 : int = 2*i
            parent2 : int = parent1 + 1

            population[parent1] = alphas[2*i] * population[parent1] + (1 - alphas[2*i]) * population[parent2]
            population[parent2] = alphas[2*i+1] * population[parent1] + (1 - alphas[2*i+1]) * population[parent2]

    def _crossoverBLX(self, population, crossover_rate, alpha : float = 0.3):
        estimated_crossovers : int = np.floor(crossover_rate * len(population) / 2).astype(int)

        for i in range(estimated_crossovers):
            parent1 : int = 2*i
            parent2 : int = parent1 + 1

            c_max : np.ndarray[float] = np.maximum(population[parent1], population[parent2])
            c_min : np.ndarray[float] = np.minimum(population[parent1], population[parent2])

            I : np.ndarray[float] = c_max - c_min

            population[parent1] = np.random.uniform(c_min - alpha * I, c_max + alpha * I)
            population[parent2] = np.random.uniform(c_min - alpha * I, c_max + alpha * I)

        population[:2*estimated_crossovers] = np.clip(population[:2*estimated_crossovers], 0, 1)


'''------------------------------------MODELO AGG------------------------------------'''

class AGG(Genetic):
    def __init__(self, crossover : str, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 0.7, improved : bool = False):
        super().__init__(crossover=crossover, seed=seed, evaluations=evaluations, population=population, mutation_rate=mutation_rate, crossover_rate=crossover_rate)
        if improved: 
            self._selection = self._best_selection
            self.params['mutation_rate'] = 2.4

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.empty(X_train.shape[1])
        self.weights = self._fit_AGG(**self.params)

    def _fit_AGG(self, max_evaluations : int = DEFAULT_MAX_EVAL, n_people : int = 50,  crossover_rate : float = 0.7, mutation_rate : float = 0.08):
        eval : int = 0
        progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

        pool = mp.Pool(psutil.cpu_count(logical=False))

        population = np.random.uniform(0, 1, (n_people, self.X_train.shape[1]))
        fitnesess = np.array(pool.map(self._fitness, population))
        eval += n_people
        progress_bar.update(n_people)
        best = np.copy(population[np.argmax(fitnesess)])
        best_fit = fitnesess[np.argmax(fitnesess)]

        # import matplotlib.pyplot as plt
        # import pandas as pd

        # df = pd.DataFrame()
        # df = pd.concat([df, pd.DataFrame({'iter': eval, 'best': best_fit, 'mean': np.mean(fitnesess), 'worst': np.min(fitnesess)}, index=[0])])

        while eval < max_evaluations:
            new_population = self._selection(population, fitnesess)
            self._crossover(new_population, crossover_rate)
            self._mutation(new_population, mutation_rate)

            fitnesess = np.array(pool.map(self._fitness, new_population))
            eval += n_people
            progress_bar.update(n_people)

            best_new = new_population[np.argmax(fitnesess)]
            best_new_fit = fitnesess[np.argmax(fitnesess)]

            if best_new_fit > best_fit:
                best = best_new
                best_fit = best_new_fit
            else:
                new_population[np.argmin(fitnesess)] = best
                fitnesess[np.argmin(fitnesess)] = best_fit

            population = new_population
        #     df = pd.concat([df, pd.DataFrame({'iter': eval, 'best': best_fit, 'mean': np.mean(fitnesess), 'worst': np.min(fitnesess)}, index=[0])])

        # plt.plot(df['iter'], df['best'], label='Best')
        # plt.plot(df['iter'], df['mean'], label='Mean')
        # plt.plot(df['iter'], df['worst'], label='Worst')
        # plt.title(f"Fitness de AGG con la función croosover {self._crossover.__name__} y la función de selección {self._selection.__name__}")
        # plt.legend()
        # plt.show()

        pool.close()

        return np.copy(best)
    

    def _selection(self, population, fitnesess):
        new_population = np.empty_like(population)

        random_indexes = np.empty((population.shape[0], 3), dtype=int)

        for i in range(random_indexes.shape[0]):
            random_indexes[i] = np.random.choice(population.shape[0], 3, replace=False)

        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(population.shape[0]), best_in_each_group]]

        return new_population

    def _best_selection(self, population, fitnesess):
        new_population = np.empty_like(population)

        random_indexes = np.random.randint(0, population.shape[0], size=(population.shape[0], 3))
        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(population.shape[0]), best_in_each_group]]

        return new_population

    def _mutation(self, population, mutation_rate):
        estimated_mutations = int(mutation_rate * population.shape[0])

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)


'''------------------------------------MODELO AGE------------------------------------'''

class AGE(Genetic):
    def __init__(self, crossover : str, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 1, improved : bool = False):
        super().__init__(crossover=crossover, seed=seed, evaluations=evaluations, population=population, mutation_rate=mutation_rate, crossover_rate=crossover_rate)
        if improved: 
            self._selection = self._best_selection
            self.params['mutation_rate'] = 0.42

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.empty(X_train.shape[1])
        self.weights = self._fit_AGE(**self.params)

    def _fit_AGE(self, max_evaluations : int = DEFAULT_MAX_EVAL, n_people : int = 50,  crossover_rate : float = 1, mutation_rate : float = 0.08):
        eval : int = 0
        progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

        pool = mp.Pool(psutil.cpu_count(logical=False))

        population = np.random.uniform(0, 1, (n_people, self.X_train.shape[1]))
        fitnesess = np.array(pool.map(self._fitness, population))
        eval += n_people
        progress_bar.update(n_people)

        pool.close()

        pool = mp.Pool(processes=2)

        # import matplotlib.pyplot as plt
        # import pandas as pd

        # df = pd.DataFrame()
        # df = pd.concat([df, pd.DataFrame({'iter': eval, 'best': np.max(fitnesess), 'mean': np.mean(fitnesess), 'worst': np.min(fitnesess)}, index=[0])])


        while eval < max_evaluations:
            childrens = self._selection(population, fitnesess)
            self._crossover(childrens, crossover_rate)
            self._mutation(childrens, mutation_rate)

            fitnesess_children = np.array(pool.map(self._fitness, childrens))
            eval += childrens.shape[0]
            progress_bar.update(childrens.shape[0])

            two_worst = np.argsort(fitnesess)[:2]

            p = np.concatenate((childrens, population[two_worst]))
            f = np.concatenate((fitnesess_children, fitnesess[two_worst]))

            best_two = np.argsort(f)[::-1][:2]

            population[two_worst] = p[best_two]
            fitnesess[two_worst] = f[best_two]

        #     df = pd.concat([df, pd.DataFrame({'iter': eval, 'best': np.max(fitnesess), 'mean': np.mean(fitnesess), 'worst': np.min(fitnesess)}, index=[0])])

        # plt.plot(df['iter'], df['best'], label='Best')
        # plt.plot(df['iter'], df['mean'], label='Mean')
        # plt.plot(df['iter'], df['worst'], label='Worst')
        # plt.title(f"Fitness de AGE con la función croosover {self._crossover.__name__} y la función de selección {self._selection.__name__}")
        # plt.legend()
        # plt.show()

        pool.close()

        return np.copy(population[np.argmax(fitnesess)])

    def _best_selection(self, population, fitnesess):
        new_population = np.empty((2, population.shape[1]))

        random_indexes = np.random.choice(population.shape[0], (2), replace=False)

        new_population = population[random_indexes]

        return new_population
    
    def _selection(self, population, fitnesess):
        new_population = np.empty((2, population.shape[1]))

        random_indexes = np.empty((2, 3), dtype=int)

        for i in range(random_indexes.shape[0]):
            random_indexes[i] = np.random.choice(population.shape[0], 3, replace=False)

        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)

        new_population = np.copy(population[random_indexes[np.arange(2), best_in_each_group]])

        return new_population

    def _mutation(self, population, mutation_rate):
        random_chances = np.random.uniform(0, 1, population.shape[0])

        population[random_chances < mutation_rate] = np.clip(np.random.normal(0, SQRT_03) + population[random_chances < mutation_rate], 0, 1)


'''------------------------------------MODELO AM------------------------------------'''

class AM(AGG):
    def __init__(self, crossover : str, bl_selection : str, seed : int = 7, evaluations : int = 15000, population : int = 50, mutation_rate : float = 0.08, crossover_rate : float = 0.7, bl_rate = 10, improved : bool = False):
        super().__init__(crossover=crossover, seed=seed, evaluations=evaluations, population=population, mutation_rate=mutation_rate, crossover_rate=crossover_rate, improved=improved)
        self._bl_selection = self._get_bl_selection_function(bl_selection)
        self.params['bl_rate'] = bl_rate

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.empty(X_train.shape[1])
        self.weights = self._fit_AM(**self.params)

    def _fit_AM(self, max_evaluations : int = DEFAULT_MAX_EVAL, n_people : int = 50,  crossover_rate : float = 0.7, mutation_rate : float = 0.08, bl_rate : int = 10):
        eval : int = 0
        n_generations : int = 0

        bl_evaluations = 2*self.X_train.shape[1]

        bl_parameters = {
            'max_iter' : bl_evaluations,
            'max_evaluations' : bl_evaluations,
            'pb' : False
        }

        progress_bar = tqdm(total=max_evaluations, position=0, leave=True, desc='Progreso', colour='red', unit='eval', smoothing=0.1)

        pool = mp.Pool(psutil.cpu_count(logical=False))

        population = np.random.uniform(0, 1, (n_people, self.X_train.shape[1]))
        fitnesess = np.array(pool.map(self._fitness, population))
        eval += n_people
        n_generations += 1
        progress_bar.update(n_people)

        best = np.copy(population[np.argmax(fitnesess)])
        best_fit = fitnesess[np.argmax(fitnesess)]

        # import matplotlib.pyplot as plt
        # import pandas as pd

        # df = pd.DataFrame()
        # df = pd.concat([df, pd.DataFrame({'iter': eval, 'best': np.max(fitnesess), 'mean': np.mean(fitnesess), 'worst': np.min(fitnesess)}, index=[0])])

        while eval < max_evaluations:
            new_population = self._selection(population, fitnesess)
            self._crossover(new_population, crossover_rate)
            self._mutation(new_population, mutation_rate)

            fitnesess = np.array(pool.map(self._fitness, new_population))
            eval += n_people
            progress_bar.update(n_people)

            if n_generations % bl_rate == 0:
                bl_selection = self._bl_selection(new_population, fitnesess)

                if max_evaluations - eval >= bl_evaluations * bl_selection.size:
                    #bl_evaluations = (max_evaluations - eval) // bl_selection.size
                
                    for index in bl_selection:
                        bl_parameters['weights'] = new_population[index]
                        bl_parameters['fitness'] = fitnesess[index]

                        new_population[index], fitnesess[index], n_eval_bl = self._fit_BL(**bl_parameters)
                        
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
        #     df = pd.concat([df, pd.DataFrame({'iter': eval, 'best': best_fit, 'mean': np.mean(fitnesess), 'worst': np.min(fitnesess)}, index=[0])])

        # plt.plot(df['iter'], df['best'], label='Best')
        # plt.plot(df['iter'], df['mean'], label='Mean')
        # plt.plot(df['iter'], df['worst'], label='Worst')
        # plt.title(f"Fitness de AM con la función selección de bl {self._bl_selection.__name__} y la función de selección {self._selection.__name__}")
        # plt.legend()
        # plt.show()

        pool.close()

        return np.copy(best)
    

    def _bl_selection(self, population, fitnesses, p : float = 0.1):
        pass
    

    def _get_bl_selection_function(self, selection : str):
        selection = selection.lower()

        match selection:
            case 'best':
                return self._select_best_chromosomes
            case 'random':
                return self._select_random_chromosomes
            case 'all':
                return self._select_all_chromosomes
            case _:
                exception = ValueError(f"La selección {selection} no está implementada.")
                raise exception

    def _select_best_chromosomes(self, population : np.ndarray[float], fitnesses : np.ndarray[float], p : float = 0.1):
        return np.argsort(fitnesses)[::-1][:int(np.ceil(p*len(population)))]

    def _select_random_chromosomes(self, population : np.ndarray[float], fitnesses : np.ndarray[float], p : float = 0.1):
        return np.random.choice(len(population), int(np.ceil(p*len(population))), replace=False)

    def _select_all_chromosomes(self, population : np.ndarray[float], fitnesses : np.ndarray[float], p : float = 1):
        return np.arange(len(population))
    

'''------------------------------------MODELO BMB------------------------------------'''

class BMB(Generic_Model):
    def __init__(self, seed : int = 7, iterations : int = 20, bl_evaluations : int = 750):
        super().__init__()
        self.seed = seed
        self.params = {
            'max_iter' : iterations,
            'max_evaluations' : bl_evaluations
        }
        np.random.seed(seed)

    def fit(self, X_train : np.ndarray, y_train : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.uniform(0, 1, X_train.shape[1])
        self.weights = self._fit_BMB(**self.params)

    def _fit_BMB(self, max_iter : int = 20, max_evaluations : int = 750):
        solutions = np.random.uniform(0, 1, (max_iter, self.X_train.shape[1]))
        fitness_solutions = np.empty(max_iter)

        bl_parameters = {
            'max_iter' : 20*self.X_train.shape[1],
            'max_evaluations' : max_evaluations,
            'pb' : False
        }

        progress_bar = tqdm(total=max_iter, position=0, leave=True, desc='Progreso', colour='red', unit='iter', smoothing=0.1)

        for i in range(max_iter):
            bl_parameters['weights'] = solutions[i]
            solutions[i], fitness_solutions[i], _ = self._fit_BL(**bl_parameters)
            progress_bar.update(1)


        best = np.argmax(fitness_solutions)

        return np.copy(solutions[best])

'''------------------------------------MODELO ILS------------------------------------'''

class ILS(Generic_Model):
    def __init__(self, fit : str, seed : int = 7, iterations : int = 20, bl_evaluations : int = 750, prob_mut : float = 0.2):
        super().__init__()
        self.seed = seed
        self.params = {
            'max_iter' : iterations,
            'max_evaluations' : bl_evaluations,
            'prob_mut' : prob_mut,
            'fit' : fit
        }
        np.random.seed(seed)

    def fit(self, X_train : np.ndarray, y_train : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.uniform(0, 1, X_train.shape[1])
        self.weights = self._fit_ILS(**self.params)

    def _fit_ILS(self, fit : str = 'BL', max_iter : int = 20, max_evaluations : int = 750, prob_mut : float = 0.2):
        actual_weights = np.random.uniform(0, 1, self.X_train.shape[1])

        progress_bar = tqdm(total=max_iter, position=0, leave=True, desc='Progreso', colour='red', unit='iter', smoothing=0.1)

        match fit:
            case 'BL':
                fit_function = self._fit_BL
            case 'ES':
                fit_function = self._fit_ES
            case '_':
                ValueError(f"La función de fit {fit} no está implementada.")

        fit_parameters = {
            'max_iter' : 20*self.X_train.shape[1],
            'max_evaluations' : max_evaluations,
            'pb' : False
        }

        fit_parameters['weights'] = actual_weights
        actual_weights, actual_fitness, _ = fit_function(**fit_parameters)
        iterations = 1
        progress_bar.update(1)

        best_weights = np.copy(actual_weights)
        best_fitness = actual_fitness

        while iterations < max_iter:
            actual_weights = self._mutate_ILS(best_weights, prob_mut)

            fit_parameters['weights'] = actual_weights
            actual_weights, actual_fitness, _ = fit_function(**fit_parameters)

            if actual_fitness > best_fitness:
                best_weights = np.copy(actual_weights)
                best_fitness = actual_fitness

            iterations += 1
            progress_bar.update(1)

        return best_weights
    
    def _mutate_ILS(self, weights : np.ndarray, prob_mut : float):
        genes_to_mutate = np.random.uniform(0, 1, weights.shape[0]) < prob_mut

        if weights.shape[0] > 3:
            while genes_to_mutate.sum() < 3:
                genes_to_mutate[np.random.randint(0, weights.shape[0])] = True


        new_weights = np.copy(weights)
        new_weights[genes_to_mutate == True] = np.clip(new_weights[genes_to_mutate == True] + np.random.uniform(-0.25, 0.25), 0, 1)

        return new_weights


'''------------------------------------MODELO ES------------------------------------'''

class ES(Generic_Model):
    def __init__(self, seed : int = 7, evaluations : int = 15000):
        super().__init__()
        self.seed = seed
        self.MAX_EVAL = evaluations
        np.random.seed(seed)

    def fit(self, X_train : np.ndarray, y_train : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.uniform(0, 1, X_train.shape[1])
        self.weights, fitness, n_eval = self._fit_ES(self.weights, max_evaluations=self.MAX_EVAL, pb=True)