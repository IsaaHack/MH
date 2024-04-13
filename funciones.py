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

def evaluationFunction(tasa_clas, tasa_red, alfa=0.75):
    return alfa*tasa_clas + (1-alfa)*tasa_red

def getEuclideanDistances(X, x):
    return np.sqrt(np.sum((X - x) ** 2, axis=1))

def getWeightedEuclideanDistances(X : np.array, W : np.array, x):
    return np.sqrt(np.sum(W * (X - x) ** 2, axis=1))

def crossoverCA(population : np.ndarray[float], crossover_rate : float):
    estimated_crossovers : int = np.floor(crossover_rate * len(population) / 2).astype(int)
    alphas = np.random.uniform(0, 1, estimated_crossovers*2)

    for i in range(estimated_crossovers):
        parent1 : int = 2*i
        parent2 : int = parent1 + 1

        population[parent1] = alphas[2*i] * population[parent1] + (1 - alphas[2*i]) * population[parent2]
        population[parent2] = alphas[2*i+1] * population[parent1] + (1 - alphas[2*i+1]) * population[parent2]

def crossoverBLX(population : np.ndarray[float], crossover_rate : float, alpha : float = 0.3):
    estimated_crossovers = np.floor(crossover_rate * len(population) / 2).astype(int)
    
    for i in range(estimated_crossovers):
        parent1 : int = 2*i
        parent2 : int = parent1 + 1

        c_max : float = np.maximum(population[parent1], population[parent2])
        c_min : float = np.minimum(population[parent1], population[parent2])

        I : float = c_max - c_min

        population[parent1] = np.random.uniform(c_min - alpha * I, c_max + alpha * I)
        population[parent2] = np.random.uniform(c_min - alpha * I, c_max + alpha * I)

    population[:2*estimated_crossovers] = np.clip(population[:2*estimated_crossovers], 0, 1)


def fiveCrossValidation(X1 : np.ndarray, X2 : np.ndarray, X3 : np.ndarray, X4 : np.ndarray, X5 : np.ndarray , y1 : np.ndarray, y2 : np.ndarray, y3 : np.ndarray, y4 : np.ndarray, y5 : np.ndarray, model_type : str, seed : int, k : int):
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

        if model_type == 'KNN':
            model = modelos.KNN(k)
        elif model_type == 'Relief':
            model = modelos.Relief()
        elif model_type == 'BL':
            model = modelos.BL(np.random.randint(0, 1000))
        elif model_type == 'AGG-AC' or model_type == 'AGG-BLX':
            model = modelos.AGG(np.random.randint(0, 1000))
        elif model_type == 'AGE-AC' or model_type == 'AGE-BLX':
            model = modelos.AGE(np.random.randint(0, 1000))
        else:
            raise ValueError("El modelo no es válido.")

        if model_type == 'AGG-AC' or model_type == 'AGE-AC':
            model.fit(X_train, y_train, crossoverCA)
        elif model_type == 'AGG-BLX' or model_type == 'AGE-BLX':
            model.fit(X_train, y_train, crossoverBLX)
        else:
            model.fit(X_train, y_train)

        # Evaluar el modelo

        tasa_red = model.red_rate()

        tasa_clas = model.clas_rate()

        fitness_train = model.fitness(clasRate=tasa_clas, redRate=tasa_red)

        accuracy = model.accuracy(X_test, y_test)

        fitness_test = evaluationFunction(accuracy, tasa_red)

        time_end : float = time.time()

        total_time : float = time_end - time_start

        peso = [i+1,','.join(map(str, model.weights))]

        pesos.append(peso)

        resultados = pd.concat([resultados, pd.DataFrame([[i+1, tasa_clas, tasa_red, fitness_train, fitness_test, accuracy, total_time]], columns=claves)], ignore_index=True)

    return resultados, pesos