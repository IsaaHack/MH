import numpy as np
import time
import subprocess
try:
    import pandas as pd
except ImportError:
    print("Pandas no está instalado. Instalando Pandas...")
    try:
        subprocess.check_call(["pip", "install", "pandas"])
        import pandas as pd
        print("Pandas se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar Pandas. Por favor, instálalo manualmente.")
import modelos

ALPHA = 0.75

class Model_Parameters:
    def __init__(self, model_type : list[str], params : list[dict], model_name : list[str]):
        self.model_type = model_type
        self.params = params
        self.model_name = model_name

def objetiveFunction(tasa_clas, tasa_red, alfa=0.75):
    return alfa*tasa_clas + (1-alfa)*tasa_red

def crossoverCA(population : np.ndarray[float], crossover_rate : float):
    estimated_crossovers : int = np.floor(crossover_rate * len(population) / 2).astype(int)
    alphas = np.random.uniform(0, 1, estimated_crossovers*2)

    for i in range(estimated_crossovers):
        parent1 : int = 2*i
        parent2 : int = parent1 + 1

        population[parent1] = alphas[2*i] * population[parent1] + (1 - alphas[2*i]) * population[parent2]
        population[parent2] = alphas[2*i+1] * population[parent1] + (1 - alphas[2*i+1]) * population[parent2]

def crossoverBLX(population : np.ndarray[float], crossover_rate : float, alpha : float = 0.3):
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

def localSearch(chromosome : np.ndarray[float], fitness : float, fitness_function, max_evaluations : int = 15000):
    evaluations : int = 0
    best_fitness : float = fitness
    best_chromosome : np.ndarray[float] = np.copy(chromosome)

    while evaluations < max_evaluations:
        mutation_order : np.ndarray[int] = np.random.permutation(len(chromosome))

        for i in mutation_order:
            neighbour : np.ndarray[float] = getNeighbour(chromosome, i)
            new_fitness : float = fitness_function(neighbour)
            evaluations += 1

            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_chromosome = np.copy(neighbour)

            if evaluations >= max_evaluations:
                break


    return best_chromosome, best_fitness, evaluations


def getNeighbour(chromosome : np.ndarray[float], i : int):
    mutation : float = np.random.normal(0, np.sqrt(0.3))
    neighbour : np.ndarray[float] = np.copy(chromosome)

    neighbour[i] = np.clip(neighbour[i] + mutation, 0, 1)

    return neighbour


def selectBestChromosomes(population : np.ndarray[float], fitness : np.ndarray[float], p : float = 0.1):
    return np.argsort(fitness)[::-1][:int(np.ceil(p*len(population)))]

def selectRandomChromosomes(population : np.ndarray[float], fitness : np.ndarray[float], p : float = 0.1):
    return np.random.choice(len(population), int(np.ceil(p*len(population))), replace=False)

def selectAllChromosomes(population : np.ndarray[float], fitness : np.ndarray[float], p : float = 1):
    return np.arange(len(population))


def fiveCrossValidation(X1 : np.ndarray, X2 : np.ndarray, X3 : np.ndarray, X4 : np.ndarray, X5 : np.ndarray , y1 : np.ndarray, y2 : np.ndarray, y3 : np.ndarray, y4 : np.ndarray, y5 : np.ndarray, model_name : str, model_params : dict, seed : int):
    claves = ['Partición', 'Tasa de clasificación', 'Tasa de reducción', 'Fitness train', 'Fitness test', 'Accuracy', 'Tiempo de ejecución']
    resultados = pd.DataFrame()
    pesos = list()
    np.random.seed(seed)
    for i in range(5):
        # Unir los conjuntos de datos
        if(i == 0):
            X_train = np.concatenate((X2, X3, X4, X5))
            y_train = np.concatenate((y2, y3, y4, y5))
            X_test = X1
            y_test = y1
        elif(i == 1):
            X_train = np.concatenate((X1, X3, X4, X5))
            y_train = np.concatenate((y1, y3, y4, y5))
            X_test = X2
            y_test = y2
        elif(i == 2):
            X_train = np.concatenate((X1, X2, X4, X5))
            y_train = np.concatenate((y1, y2, y4, y5))
            X_test = X3
            y_test = y3
        elif(i == 3):
            X_train = np.concatenate((X1, X2, X3, X5))
            y_train = np.concatenate((y1, y2, y3, y5))
            X_test = X4
            y_test = y4
        elif(i == 4):
            X_train = np.concatenate((X1, X2, X3, X4))
            y_train = np.concatenate((y1, y2, y3, y4))
            X_test = X5
            y_test = y5

        # Entrenar el modelo

        time_start : float = time.time()

        match model_name:
            case 'KNN':
                model = modelos.KNN(**model_params)
            case 'Relief':
                model = modelos.Relief(**model_params)
            case 'BL':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.BL(**model_params)
            case 'AGG':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.AGG(**model_params)
            case 'AGE':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.AGE(**model_params)
            case 'AM':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.AM(**model_params)
            case _:
                print("Modelo no válido.")
                exit()

        # Entrenar el modelo

        model.fit(X_train, y_train)

        # Evaluar el modelo

        tasa_red = model.red_rate()

        tasa_clas = model.clas_rate()

        fitness_train = model.fitness(clasRate=tasa_clas, redRate=tasa_red)

        accuracy = model.accuracy(X_test, y_test)

        fitness_test = objetiveFunction(accuracy, tasa_red, ALPHA)

        time_end : float = time.time()

        total_time : float = time_end - time_start

        peso = [i+1,','.join(map(str, model.weights))]

        pesos.append(peso)

        resultados = pd.concat([resultados, pd.DataFrame([[i+1, tasa_clas, tasa_red, fitness_train, fitness_test, accuracy, total_time]], columns=claves)], ignore_index=True)

    return resultados, pesos