import numpy as np

def n_filas(X):
    return X.shape[0]

def n_columnas(X):
    return X.shape[1]

def ordenarPorDistancia(distancias, eje):
    return np.argsort(distancias, axis=eje)


def distanciaEuclidiana(x1, x2, pesos):
    return np.sqrt(np.sum((x1 - x2) ** 2)*pesos)

def matrizCuadradaDistancias(X, pesos):
    distancias = np.zeros((X.shape[0], X.shape[0]))
    
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

def fitness(X, y, pesos, alfa):
    return tasa_clasificacion(X, y, pesos) * alfa + tasa_reduccion(pesos) * (1 - alfa)

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

    return tasa_clasificacion / n_filas(X)

def tasa_reduccion(pesos):
    cont = 0
    for p in pesos:
        if p <= 0.1:
            cont += 1

    return cont / len(pesos)


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
    pesos = np.zeros(n_columnas(X))
    
    # Calcular matriz de distancias
    distancias = matrizCuadradaDistancias(X, pesos)
    # Ordenar indices de distancias
    indices_ord = ordenarPorDistancia(distancias, eje='fila')

    for i in range(n_filas(X)):
        amigo_cercano = -1
        
        # Filtrar amigos y quitamos los que tengan distancia 0
        amigos = filtrarAmigos(i, indices_ord, y)
        amigos = quitarDistanciaCero(i, amigos, distancias)

        # Si hay amigos, el amigo cercano es el primero
        if len(amigos) > 0:
            amigo_cercano = amigos[0]

        # Filtrar enemigos
        enemigo_cercano = filtrarEnemigos(i, indices_ord, y)

        # Si hay amigos, actualizamos los pesos
        if amigo_cercano != -1:
            pesos += abs(X[i] - X[amigo_cercano]) - abs(X[i] - X[enemigo_cercano])

    # Normalizar pesos y ponerlos en el rango [0, 1]
    maximo  = valor_maximo(pesos)

    for p in pesos:
        p = max(p/maximo, 0)

    return pesos



def fit_BL(X, y, max_evaluaciones):
    # Incializar pesos con una distribución uniforme con el tamaño de columnas de X
    pesos = np.random.uniform(0, 1, n_columnas(X))

    # Maximo de iteraciones de veces en la que toleramos no mejorar
    MAX_ITER = 20*len(X)

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