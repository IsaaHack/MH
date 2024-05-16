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

def safeRandomMultiprocessing(seed : int, function : callable, params : dict):
    np.random.seed(seed)
    return function(**params)

def objetiveFunction(tasa_clas, tasa_red, alfa=0.75):
    return alfa*tasa_clas + (1-alfa)*tasa_red


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
            case 'BMB':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.BMB(**model_params)
            case 'ILS':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.ILS(**model_params)
            case 'ES':
                model_params['seed'] = np.random.randint(0, 1000)
                model = modelos.ES(**model_params)
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