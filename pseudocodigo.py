import numpy as np
import time

def sqrt(x):
    return np.sqrt(x)

def n_filas(X):
    return X.shape[0]

def n_columnas(X):
    return X.shape[1]

def leer_arff(path):
    pass

def escaladoMinmax(X):
    minimo = np.min(X, axis=0)
    maximo = np.max(X, axis=0)

    return (X - minimo) / (maximo - minimo)

def ordenarPorDistancia(distancias, eje):
    return np.argsort(distancias, axis=eje)


def distanciaEuclidiana(x1, x2, pesos):
    suma = 0
    for i in range(n_columnas(x1)):
        # Si el peso es menor o igual a 0.1, no se tiene en cuenta
        if pesos[i] <= 0.1:
            continue

        diff = x1[i] - x2[i]
        diff = diff ** 2
        suma += diff * pesos[i]

    return sqrt(suma)

def matrizCuadradaDistancias(X, pesos):
    distancias = np.zeros((n_filas(X), n_filas(X)))
    
    for i in range(n_filas(X)):
        for j in range(n_filas(X)):
            distancias[i][j] = distanciaEuclidiana(X[i], X[j], pesos)
    
    return distancias

def filtrarAmigos(i, indices_ord, y):
    amigos = []
    
    for j in range(n_filas(indices_ord)):
        if y[indices_ord[i]] == y[indices_ord[j]]:
            amigos.append(indices_ord[j])
    
    return amigos

def filtrarEnemigos(i, indices_ord, y):
    enemigos = []
    
    for j in range(n_filas(indices_ord)):
        if y[indices_ord[i]] != y[indices_ord[j]]:
            enemigos.append(indices_ord[j])
    
    return enemigos

def quitarDistanciaCero(i, amigos, distancias):
    for j in range(amigos.size()):
        if distancias[i][amigos[j]] == 0:
            amigos.pop(j)

    return amigos

def valor_maximo(pesos):
    maximo = pesos[0]
    for i in range(len(pesos)):
        if pesos[i] > maximo:
            maximo = pesos[i]

    return maximo

def fitness(X, y, pesos, alfa=0.75):
    return tasa_clasificacion(X, y, pesos) * alfa + tasa_reduccion(pesos) * (1 - alfa)

def fitness2(aciertos, tasa_red, alfa=0.75):
    return alfa*aciertos + (1-alfa)*tasa_red

def fitness(X_train, y_train, X_test, y_test, pesos, alfa=0.75):
    predicciones = clasificador1NN(X_train, y_train, X_test, pesos)
    aciertos = accuracy(y_test, predicciones)
    tasa_red = tasa_reduccion(pesos)
    return alfa*aciertos + (1-alfa)*tasa_red

def tasa_clasificacion(X, y, pesos):
    # Calcular matriz de distancias
    distancias = matrizCuadradaDistancias(X, pesos)
    # Ordenar indices de distancias
    indices_ord = ordenarPorDistancia(distancias, eje='fila')

    # Inicializar tasa de clasificacion
    tasa_clasificacion = 0

    for i in range(n_filas(X)):
        # Como los indices_ord estan ordenados, el mas cercano es el primero
        # Hay que ignorar el indice 0 porque es el mismo
        mas_cercano = indices_ord[i][1]

        if y[i] == y[mas_cercano]:
            tasa_clasificacion += 1

    return 100*(tasa_clasificacion / n_filas(X))

def tasa_reduccion(pesos):
    cont = 0
    for p in pesos:
        if p >= 0.1:
            cont += 1

    return 100*(cont / len(pesos))


def obtenerVecino(pesos, i):
    #Obtener una mutacion aleatoria de la distribucion normal de media 0 y varianza 0.3
    mutacion = np.random.normal(0, np.sqrt(0.3))
    vecino = pesos.copy()

    # Aplicar la mutacion
    vecino[i] += mutacion

    # Normalizar el peso cambiado
    vecino[i] = max(vecino[i], 0)
    vecino[i] = min(vecino[i], 1)

    return vecino


def fit_Relief(X, y):
    # Inicializar pesos a 0 con la cantidad de columnas de X
    pesos = []

    for i in range(n_columnas(X)):
        pesos.append(0)
    
    # Calcular matriz de distancias
    distancias = matrizCuadradaDistancias(X, pesos)
    distancias[distancias == 0] = np.inf
    # Ordenar indices de distancias
    indices_ord = ordenarPorDistancia(distancias, eje='fila')

    for i in range(n_filas(X)):
        amigo_cercano = -1
        
        # Filtrar amigos y quitamos los que tengan distancia 0
        amigos = filtrarAmigos(i, indices_ord, y)

        # Si hay amigos, el amigo cercano es el primero
        if len(amigos) > 0:
            amigo_cercano = amigos[0]

        # Filtrar enemigos
        enemigo_cercano = filtrarEnemigos(i, indices_ord, y)

        # Si hay amigos, actualizamos los pesos
        if amigo_cercano != -1 and distancias[i][amigo_cercano] != np.inf:
            pesos += abs(X[i] - X[amigo_cercano]) - abs(X[i] - X[enemigo_cercano])

    # Normalizar pesos y ponerlos en el rango [0, 1]
    maximo  = valor_maximo(pesos)

    for p in pesos:
        p = max(p/maximo, 0)

    return pesos



def fit_BL(X, y, max_evaluaciones, semilla=7):
    # Semilla para inicializar pesos aleatorios
    np.random.seed(semilla)

    # Incializar pesos con una distribución uniforme con el tamaño de columnas de X
    pesos = np.random.uniform(0, 1, n_columnas(X))

    # Maximo de iteraciones de veces en la que toleramos no mejorar
    MAX_ITER = 20*n_columnas(X)

    # Alfa para el fitness
    ALFA = 0.75

    # Evaluacion actual
    fitness_actual = fitness(X, y, pesos, ALFA)

    n_evaluaciones = 0
    iteraciones_sin_mejora = 0

    while n_evaluaciones < max_evaluaciones and iteraciones_sin_mejora < MAX_ITER:
        # Obtener un orden de mutacion aleatorio
        orden_mutacion = np.random.permutation(len(pesos))

        for mut in orden_mutacion:
            # Obtener un vecino
            vecino = obtenerVecino(pesos, mut)

            # Calcular fitness del vecino
            fitness_vecino = fitness(X, y, vecino, ALFA)
            n_evaluaciones += 1

            # Si el vecino es mejor, actualizamos pesos
            if fitness_vecino > fitness_actual:
                pesos = vecino
                fitness_actual = fitness_vecino
                iteraciones_sin_mejora = 0
            else: # Si no, aumentamos el contador de iteraciones sin mejora
                iteraciones_sin_mejora += 1

    return pesos

def clasificador1NN(X_train, y_test, X_test, pesos=None, k=1):
    # Inicializar pesos a 1 si no se especifican simulando un KNN normal
    if pesos is None:
        pesos = []
        for i in range(n_columnas(X_train)):
            pesos.append(1)

    predictions = np.array(n_columnas(X_test), dtype=str)

    # Calcular matriz de distancias
    for i in range(n_filas(X_test)):
        distancias = np.zeros(n_filas(X_train))
        for j in range(n_filas(X_train)):
            distancias[j] = distanciaEuclidiana(X_test[i], X_train[j], pesos)

        indices_ord = ordenarPorDistancia(distancias, eje='fila')

        # Si elegimos el numero 1, el vecino más cercano es el segundo porque el primero es el mismo
        indice_cercano = indices_ord[1]

        # Asignar la clase del vecino más cercano
        clase = y_test[indice_cercano]

        # Asignar la clase al conjunto de predicciones
        predictions[i] = clase

    return predictions

def accuracy(y_true, y_pred):
    correctos = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correctos += 1

    return correctos / len(y_true)


def fiveCrossValidation(X1, X2, X3, X4, X5, y1, y2, y3, y4, y5, model_type, seed=7, k=1):
    resultados = []
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

        time_start = time.time()

        pesos = np.array(n_columnas(X_train))

        if model_type == 'KNN':
            pesos = np.ones(n_columnas(X_train))
        elif model_type == 'Relief':
            pesos = fit_Relief(X_train, y_train)
        elif model_type == 'BL':
            pesos = fit_BL(X_train, y_train, max_evaluaciones=15000, semilla=np.random.randint(0, 1000))
        else:
            raise ValueError("El modelo no es válido.")

        # Evaluar el modelo

        tasa_red = tasa_reduccion(pesos)

        tasa_clas = tasa_clasificacion(X_train, y_train, pesos)

        fitness_train = fitness(X_train, y_train, pesos, 0.75)

        acierto = accuracy(y_test, clasificador1NN(X_train, y_test, X_test, pesos, k))

        fitness_test = fitness2(acierto, tasa_red, 0.75)

        time_end = time.time()

        total_time = time_end - time_start

        resultados.append((tasa_red, tasa_clas, fitness_train, acierto, fitness_test, total_time))

    return resultados

def main():
    print('Introduce el nombre del conjunto de datos:')
    cadena = input()

    todos_los_modelos = ['KNN', 'Relief', 'BL', 'ALL']
    print('Introduce el modelo a utilizar [KNN, Relief, BL, ALL]:')
    model_type = input().upper()

    if model_type == 'KNN' or model_type == 'ALL':
        print('Introduce el valor de k:')
        k = int(input())

    if model_type == 'BL' or model_type == 'ALL':
        print('Introduce el valor de la semilla [DEFAULT=7]:')
        seed_i = input()
        if seed_i == '':
            seed = 7
        else:
            seed = int(seed_i)

    # Cargar los 5 conjuntos de datos
    data1 = leer_arff('./data/'+cadena+'_1.arff')
    data2 = leer_arff('./data/'+cadena+'_2.arff')
    data3 = leer_arff('./data/'+cadena+'_3.arff')
    data4 = leer_arff('./data/'+cadena+'_4.arff')
    data5 = leer_arff('./data/'+cadena+'_5.arff')

    # Separar los datos en características y etiquetas
    X1 = np.array(data1[:, :-1], dtype=float)
    y1 = data1[:, -1]

    X2 = np.array(data2[:, :-1], dtype=float)
    y2 = data2[:, -1]

    X3 = np.array(data3[:, :-1], dtype=float)
    y3 = data3[:, -1]

    X4 = np.array(data4[:, :-1], dtype=float)
    y4 = data4[:, -1]

    X5 = np.array(data5[:, :-1], dtype=float)
    y5 = data5[:, -1]

    # Normalizar los datos

    X = np.concatenate((X1, X2, X3, X4, X5), axis=0)

    X = escaladoMinmax(X)

    X1 = X[:n_filas(X1)]
    X2 = X[n_filas(X1):n_filas(X1)+n_filas(X2)]
    X3 = X[n_filas(X1)+n_filas(X2):n_filas(X1)+n_filas(X2)+n_filas(X3)]
    X4 = X[n_filas(X1)+n_filas(X2)+n_filas(X3):n_filas(X1)+n_filas(X2)+n_filas(X3)+n_filas(X4)]
    X5 = X[n_filas(X1)+n_filas(X2)+n_filas(X3)+n_filas(X4):]

    # Five cross validation
    modelos_a_utilizar = [model_type] if model_type != 'ALL' else todos_los_modelos

    for modelo in modelos_a_utilizar:
        resultados = fiveCrossValidation(X1, X2, X3, X4, X5, y1, y2, y3, y4, y5, modelo, seed, k)

        print(f'Modelo: {modelo}')

        for res in resultados:
            print(res)

        