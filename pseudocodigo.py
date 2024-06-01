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

def generar_poblacion_inicial(n_poblacion, n_caracteristicas):
    # Inicializar la poblacion con una matriz de n_poblacion x n_caracteristicas de pesos aleatorios de forma uniforme
    poblacion = np.empty((n_poblacion, n_caracteristicas))

    for i in range(n_poblacion):
        for j in range(n_caracteristicas):
            poblacion[i][j] = np.random.uniform(0, 1)


def fit_AGG(X, y, max_evaluaciones, tipo_cruce, semilla=7, n_poblacion = 50, prob_cruce = 0.7, prob_mutacion = 0.08):
    # Semilla para inicializar pesos aleatorios
    np.random.seed(semilla)
    evaluaciones = 0

    # Inicializar la poblacion con pesos aleatorios de tamaño poblacion x columnas de X
    poblacion = generar_poblacion_inicial(n_poblacion, n_columnas(X))

    # Initializar fitness de la poblacion
    fitness_poblacion = np.empty_like(poblacion)

    for i in range(n_filas(poblacion)):
        fitness_poblacion[i] = fitness(X, y, poblacion[i])
        evaluaciones += 1

    # Obtener el mejor individuo
    fitness_mejor_individuo = np.max(fitness_poblacion)
    mejor_individuo = poblacion[np.argmax(fitness_poblacion)]

    # Mientras no se alcance el maximo de evaluaciones
    while evaluaciones < max_evaluaciones:
        # Seleccionar padres
        padres = seleccionar_padres_AGG(poblacion, fitness_poblacion)

        # Cruzar padres
        hijos = cruce(padres, prob_cruce, tipo_cruce)

        # Mutar hijos
        mutar_AGG(hijos, prob_mutacion)

        # Evaluar hijos
        for i in range(n_filas(hijos)):
            fitness_poblacion[i] = fitness(X, y, hijos[i])
            evaluaciones += 1

        #Reemplazar los peores individuos

        # Mejor de la nueva generacion
        mejor_hijo = np.argmax(fitness_poblacion)
        fitness_mejor_hijo = np.max(fitness_poblacion)

        # Si el mejor hijo es mejor que el mejor individuo, lo actualizamos
        if fitness_mejor_hijo > fitness_mejor_individuo:
            mejor_individuo = hijos[mejor_hijo]
            fitness_mejor_individuo = fitness_mejor_hijo
        else:
            # Si no, reemplazamos el peor hijo por el mejor individuo
            peor_hijo = np.argmin(fitness_poblacion)
            hijos[peor_hijo] = mejor_individuo
            fitness_poblacion[peor_hijo] = fitness_mejor_individuo

        # Actualizamos la poblacion
        poblacion = hijos

    # Devolvemos el mejor individuo
    return mejor_individuo

def seleccionar_padres_AGG(poblacion, fitness_poblacion):
    # Seleccionar padres por torneo
    padres = []

    for i in range(n_filas(poblacion)):
        # Seleccionar tres individuos aleatorios de la poblacion distintos
        indices_aleatorios = np.random.choice(range(0, n_filas(poblacion)), size=3)

        # Seleccionar el mejor de los tres
        mejor = np.argmax(fitness_poblacion[indices_aleatorios])

        # Añadir el mejor a la lista de padres
        padres.append(poblacion[indices_aleatorios[mejor]])

    return padres

def cruce(padres, prob_cruce, tipo_cruce):
    match tipo_cruce:
        case 'AC':
            return cruce_CA(padres, prob_cruce)
        case 'BLX':
            return cruce_BLX(padres, prob_cruce)
        case _:
            raise ValueError("Tipo de cruce no válido.")
        

def cruce_CA(padres, prob_cruce):
    # Calcular cruces esperados
    cruces_esperados = int(prob_cruce * n_filas(padres) / 2)
    # Generar alphas aleatorios
    alphas = np.random.uniform(0, 1, size=cruces_esperados*2)

    hijos = []

    for i in range(cruces_esperados):
        hijo1 = padres[i*2] * alphas[i] + padres[i*2+1] * (1 - alphas[i])
        hijo2 = padres[i*2+1] * alphas[i+1] + padres[i*2] * (1 - alphas[i+1])

        hijos.append(hijo1)
        hijos.append(hijo2)

    return hijos

def cruce_BLX(padres, prob_cruce):
    # Calcular cruces esperados
    cruces_esperados = int(prob_cruce * n_filas(padres) / 2)
    # Alphas para el BLX-0.3
    ALPHA = 0.3
    
    hijos = []

    for i in range(cruces_esperados):
        hijo1 = np.empty(shape=n_columnas(padres))
        hijo2 = np.empty(shape=n_columnas(padres))

        for j in range(n_columnas(padres)):
            # Calcular minimo y maximo
            minimo = min(padres[i*2][j], padres[i*2+1][j])
            maximo = max(padres[i*2][j], padres[i*2+1][j])

            # Calcular rango
            I = maximo - minimo

            # Formulas de BLX-0.3
            # aleatorio( minimo - ALPHA*I, maximo + ALPHA*I)

            # Calcular valores de los hijos
            hijo1[j] = np.random.uniform(minimo - ALPHA*I, maximo + ALPHA*I)
            hijo2[j] = np.random.uniform(minimo - ALPHA*I, maximo + ALPHA*I)

        # Añadir hijos a la lista
        hijos.append(hijo1)
        hijos.append(hijo2)

    return hijos

def mutar_AGG(hijos, prob_mutacion):
    mutaciones_esperadas = int(prob_mutacion * n_filas(hijos))

    for i in range(mutaciones_esperadas):
        # Seleccionar un hijo aleatorio
        hijo = hijos[np.random.randint(0, n_filas(hijos))]

        # Seleccionar un gen aleatorio
        gen = np.random.randint(0, n_columnas(hijo))

        # Mutar gen
        hijo[gen] = np.random.uniform(0, 1)

def fit_AGE(X, y, max_evaluaciones, tipo_cruce, semilla=7, n_poblacion = 50, prob_cruce = 1, prob_mutacion = 0.08):
    # Semilla para inicializar pesos aleatorios
    np.random.seed(semilla)
    evaluaciones = 0

    # Inicializar la poblacion con pesos aleatorios de tamaño poblacion x columnas de X
    poblacion = generar_poblacion_inicial(n_poblacion, n_columnas(X))

    # Initializar fitness de la poblacion
    fitness_poblacion = np.empty_like(poblacion)

    for i in range(n_filas(poblacion)):
        fitness_poblacion[i] = fitness(X, y, poblacion[i])
        evaluaciones += 1

    # Mientras no se alcance el maximo de evaluaciones
    while evaluaciones < max_evaluaciones:
        # Seleccionar 2 padres
        padres = seleccionar_padres_AGE(poblacion, fitness_poblacion)

        # Cruzar padres
        hijos = cruce(padres, prob_cruce, tipo_cruce)

        # Mutar hijos
        mutar_AGE(hijos, prob_mutacion)

        # Evaluar hijos en este caso son solo 2
        fitness_hijos = np.empty_like(hijos)
        for i in range(n_filas(hijos)):
            fitness_hijos[i] = fitness(X, y, hijos[i])
            evaluaciones += 1

        # Reemplazar los peores individuos
        # Los hijos compiten por entrar en la poblacion

        # Obtener los dos peores individuos
        peores = np.argsort(fitness_poblacion, order='desc')[:2]

        competidores = np.concatenate((hijos, poblacion[peores]))
        fitness_competidores = np.concatenate((fitness_hijos, fitness_poblacion[peores]))

        # Obtener los dos mejores individuos
        mejores = np.argsort(fitness_competidores, order='desc')[:2]

        # Reemplazar los peores por los mejores
        poblacion[peores] = competidores[mejores]
        fitness_poblacion[peores] = fitness_competidores[mejores]

    # Devolvemos el mejor individuo
    mejor_individuo = poblacion[np.argmax(fitness_poblacion)]
    return mejor_individuo


def seleccionar_padres_AGE(poblacion, fitness_poblacion):
    # Seleccionar padres por torneo
    padres = []

    # Solo se seleccionan dos padres
    for i in range(2):
        # Seleccionar tres individuos aleatorios de la poblacion distintos
        indices_aleatorios = np.random.choice(range(0, n_filas(poblacion)), size=3)

        # Seleccionar el mejor de los tres
        mejor = np.argmax(fitness_poblacion[indices_aleatorios])

        # Añadir el mejor a la lista de padres
        padres.append(poblacion[indices_aleatorios[mejor]])

    return padres

def mutar_AGE(hijos, prob_mutacion):
    for i in range(n_filas(hijos)):
        for j in range(n_columnas(hijos[i])):
            if np.random.uniform(0, 1) < prob_mutacion:
                hijos[i][j] += np.random.normal(0, np.sqrt(0.3))
                hijos[i][j] = max(hijos[i][j], 0)
                hijos[i][j] = min(hijos[i][j], 1)

def fit_AM(X, y, max_evaluaciones, tipo_seleccion_bl, semilla=7, n_poblacion = 50, prob_cruce = 0.7, prob_mutacion = 0.08):
    # Semilla para inicializar pesos aleatorios
    np.random.seed(semilla)
    evaluaciones = 0
    generaciones = 0

    max_iter_bl = 2*n_columnas(X)

    # Inicializar la poblacion con pesos aleatorios de tamaño poblacion x columnas de X
    poblacion = generar_poblacion_inicial(n_poblacion, n_columnas(X))

    # Initializar fitness de la poblacion
    fitness_poblacion = np.empty_like(poblacion)

    for i in range(n_filas(poblacion)):
        fitness_poblacion[i] = fitness(X, y, poblacion[i])
        evaluaciones += 1

    generaciones += 1

    # Obtener el mejor individuo
    fitness_mejor_individuo = np.max(fitness_poblacion)
    mejor_individuo = poblacion[np.argmax(fitness_poblacion)]

    # Mientras no se alcance el maximo de evaluaciones
    while evaluaciones < max_evaluaciones:
        # Seleccionar padres
        padres = seleccionar_padres_AGG(poblacion, fitness_poblacion)

        # Cruzar padres
        hijos = cruce(padres, prob_cruce, tipo_cruce='BLX')

        # Mutar hijos
        mutar_AGG(hijos, prob_mutacion)

        # Evaluar hijos
        for i in range(n_filas(hijos)):
            fitness_poblacion[i] = fitness(X, y, hijos[i])
            evaluaciones += 1

        # Hacemos una seleccion de individuos de bl cada 10 generaciones
        if generaciones % 10 == 0:
            # Seleccionar individuos de la poblacion
            seleccionados = seleccion_BL(poblacion, fitness_poblacion, tipo_seleccion_bl)

            # Si hay sufitientes evaluaciones, aplicamos busqueda local
            if max_evaluaciones - evaluaciones < max_iter_bl * n_filas(seleccionados):
                for i in range(n_filas(seleccionados)):
                    seleccionados[i], fitness_poblacion[i], n_eval_bl = BL(X, y, seleccionados[i], fitness_poblacion[i], max_iter_bl)
                    evaluaciones += n_eval_bl


        #Reemplazar los peores individuos

        # Mejor de la nueva generacion
        mejor_hijo = np.argmax(fitness_poblacion)
        fitness_mejor_hijo = np.max(fitness_poblacion)

        # Si el mejor hijo es mejor que el mejor individuo, lo actualizamos
        if fitness_mejor_hijo > fitness_mejor_individuo:
            mejor_individuo = hijos[mejor_hijo]
            fitness_mejor_individuo = fitness_mejor_hijo
        else:
            # Si no, reemplazamos el peor hijo por el mejor individuo
            peor_hijo = np.argmin(fitness_poblacion)
            hijos[peor_hijo] = mejor_individuo
            fitness_poblacion[peor_hijo] = fitness_mejor_individuo

        # Actualizamos la poblacion
        poblacion = hijos
        generaciones += 1

    # Devolvemos el mejor individuo
    return mejor_individuo

def BL(X, y, individuo, fitness, max_eval):
    # Inicializar el numero de evaluaciones
    n_evaluaciones = 0
    fitness_individuo = fitness

    while n_evaluaciones < max_eval:
        # Obtener un orden de mutacion aleatorio
        orden_mutacion = np.random.permutation(len(individuo))

        for mut in orden_mutacion:
            # Obtener un vecino
            vecino = obtenerVecino(individuo, mut)

            # Calcular fitness del vecino
            fitness_vecino = fitness(X, y, vecino)
            n_evaluaciones += 1

            # Si el vecino es mejor, actualizamos pesos
            if fitness_vecino > fitness_individuo:
                individuo = vecino
                fitness_individuo = fitness_vecino

            # Si se alcanza el maximo de evaluaciones, salimos
            if n_evaluaciones >= max_eval:
                break

    return individuo, fitness_individuo, n_evaluaciones

def seleccion_BL(poblacion, fitness_poblacion, tipo_seleccion_bl):
    match tipo_seleccion_bl:
        case 'Mejores':
            # Seleccionar el 10% de los mejores individuos
            p = 0.1
            return np.argsort(fitness_poblacion, order='asc')[0:int(p*n_filas(poblacion))]
        case 'Aleatorios':
            # Seleccionar el 10% de los individuos aleatorios
            p = 0.1
            return np.random.choice(poblacion, size=int(p*n_filas(poblacion)))
        case 'Todos':
            # Seleccionar todos los individuos
            return range(n_filas(poblacion))
        case _:
            raise ValueError("Tipo de seleccion no válido.")
        
def fit_ES(X, y, individuo, max_eval):
    # Inicializar el numero de evaluaciones
    n_evaluaciones = 0

    # Inicializar el fitness del individuo
    fitness_individuo = fitness(X, y, individuo)

    # Peso actual
    actual = individuo
    fitness_actual = fitness_individuo

    # Guardamos el mejor individuo
    mejor_individuo = individuo
    fitness_mejor_individuo = fitness_individuo

    temperatura = temperatura_inicial(sigma=0.3, mu=0.1, eval=fitness_actual)
    temperatura_final = 10e-3

    # Inicializar maximo de vecinos, exitos y enfriamientos
    max_vecinos = 10*n_columnas(individuo)
    max_exitos = max_vecinos // 10
    max_enfriamientos = max_eval // max_vecinos

    beta = (temperatura - temperatura_final) / (max_enfriamientos * temperatura * temperatura_final)

    # Varible para saber si ha habido exito
    exito = True

    while n_evaluaciones < max_eval and exito:
        n_vecinos = 0
        n_exitos = 0

        while n_evaluaciones < max_eval and n_vecinos < max_vecinos and n_exitos < max_exitos:
            # Obetener el gen a mutar
            gen = np.random.randint(0, n_columnas(individuo))

            # Obtener un vecino
            vecino = obtenerVecino(actual, gen)

            # Calcular fitness del vecino
            fitness_vecino = fitness(X, y, vecino)
            n_evaluaciones += 1
            n_vecinos += 1

            delta = fitness_actual - fitness_vecino

            # Si el vecino es mejor,o se decide por probabilidad,empeorar se actualiza el actual
            if delta < 0 or np.random.uniform(0, 1) <= np.exp(delta/temperatura):
                actual = vecino
                fitness_actual = fitness_vecino
                n_exitos += 1

                # Si el vecino es mejor que el mejor individuo, lo actualizamos
                if fitness_vecino > fitness_mejor_individuo:
                    mejor_individuo = vecino
                    fitness_mejor_individuo = fitness_vecino

        # Enfriar la temperatura
        temperatura = enfriar(temperatura, beta)
        # Comprobar si ha habido exito
        exito = n_exitos > 0

    return mejor_individuo, fitness_mejor_individuo, n_evaluaciones

def temperatura_inicial(sigma, mu, eval):
    # La ecuacion es mu*eval/-ln(sigma)
    return mu*eval/-np.log(sigma)

def enfriar(temperatura, beta):
    # Esquema de enfriamiento Cauchy modificado
    return temperatura / (1 + beta*temperatura)

def fit_BMB(X, y, max_iter, eval_bl, semilla=7):
    # Semilla para inicializar pesos aleatorios
    np.random.seed(semilla)

    # Inicializar pesos con una distribución uniforme con el tamaño de columnas de X
    soluciones_iniciales = generar_poblacion_inicial(max_iter, n_columnas(X))
    fitness_soluciones = np.empty(max_iter)

    # Mejorar las soluciones iniciales con busqueda local
    for i in range(max_iter):
        soluciones_iniciales[i], fitness_soluciones[i], _ = BL(X, y, soluciones_iniciales[i], eval_bl)

    # Obtener la mejor solucion
    mejor_solucion = soluciones_iniciales[np.argmax(fitness_soluciones)]

    return mejor_solucion

def fit_ILS(X, y, tipo_fit, max_iter, eval_bl, prob_mutacion, semilla=7):
    # Semilla para inicializar pesos aleatorios
    np.random.seed(semilla)

    # Elegir el la funcion de fit que se va a utilizar, si BL o ES
    match tipo_fit:
        case 'BL':
            fit = BL
        case 'ES':
            fit = fit_ES
        case _:
            raise ValueError("Tipo de fit no válido.")

    # Inicializar pesos con una distribución uniforme con el tamaño de columnas de X
    solucion_actual = np.random.uniform(0, 1, n_columnas(X))

    # Calcular el fitness de la solucion actual
    solucion_actual, fitness_actual, _ = fit(X, y, solucion_actual, eval_bl)
    iteraciones = 1

    # Almacenar la mejor solucion
    mejor_solucion = solucion_actual
    mejor_fitness = fitness_actual

    while iteraciones < max_iter:
        # Mutar la mejor solucion
        solucion_actual = mutacion_ILS(mejor_solucion, prob_mutacion)

        #Mejorar la solucion mutada
        solucion_actual, fitness_actual, _ = fit(X, y, solucion_actual, eval_bl)
        iteraciones += 1

        # Si la solucion actual es mejor que la mejor, la actualizamos
        if fitness_actual > mejor_fitness:
            mejor_solucion = solucion_actual
            mejor_fitness = fitness_actual

    return mejor_solucion

def mutacion_ILS(solucion, prob_mutacion):
    # Elelgir genes aleatorios a mutar
    genes = np.random.uniform(0, 1, n_columnas(solucion)) < prob_mutacion

    # HSi hay almenos 3 genes a mutar, necesitamos al menos 3 genes que mutar
    if n_columnas(solucion) > 3:
        while np.sum(genes) < 3:
            genes[np.random.randint(0, n_columnas(solucion))] = True

    # Mutar los genes
    nueva_solucion = solucion.copy()

    for i in range(n_columnas(solucion)):
        if genes[i]:
            # Variar el gen en un rango de [-0.25, 0.25]
            nueva_solucion[i] = nueva_solucion[i] + np.random.uniform(-0.25, 0.25)

            # Poner el gen en el rango [0, 1]
            nueva_solucion[i] = max(nueva_solucion[i], 0)
            nueva_solucion[i] = min(nueva_solucion[i], 1)

    return nueva_solucion


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

        