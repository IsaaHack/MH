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

try:
    import arff
except ImportError:
    print("arff no está instalado. Instalando arff...")
    try:
        subprocess.check_call(["pip", "install", "liac-arff"])
        import arff
        print("arff se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar arff. Por favor, instálalo manualmente.")

import time
import modelos
import funciones

print('Introduce la base de datos a utilizar [Opciones: BreastCancer[DEFAULT][1], Ecoli[2], Parkinson[3]]:')
database = input()
cadena = 'breast-cancer'

if database == 'BreastCancer' or database == '' or database == '1':
    cadena = 'breast-cancer'
elif database == 'Ecoli' or database == '2':
    cadena = 'ecoli'
elif database == 'Parkinson' or database == '3':
    cadena = 'parkinsons'
elif database != '':
    print('Base de datos no válida')
    exit()

print('Elige el modelo a utilizar [Opciones: KNN[DEFAULT][1], Relief[2], BL[3]):')
model_type = input()

if model_type == 'KNN' or model_type == '' or model_type == '1':
    model_type = 'KNN'
elif model_type == 'Relief' or model_type == '2':
    model_type = 'Relief'
elif model_type != 'BL' or model_type != '3':
    model_type = 'BL'
elif model_type != '':
    print('Modelo no válido')
    exit()

k = 1
seed = 7
if(model_type == 'KNN'):
    print('Introduce el valor de k [DEFAULT=1]:')
    k = input()
    if k == '':
        k = 1
    elif k != '':
        k = int(k)
    
    if k < 1:
        print('Valor de k no válido')
        exit()
elif model_type == 'BL':
    print('Introduce el valor de la semilla [DEFAULT=7]:')
    seed_i = input()
    if seed_i == '':
        seed = 7
    else:
        seed = int(seed_i)

time_total_start = time.time()

# Cargar los 5 conjuntos de datos
data1 = arff.load(open('./data/'+cadena+'_1.arff', 'r'))
feature_names = [i[0] for i in data1['attributes']]
data1 = np.array(data1['data'])

data2 = arff.load(open('./data/'+cadena+'_2.arff', 'r'))
data2 = np.array(data2['data'])

data3 = arff.load(open('./data/'+cadena+'_3.arff', 'r'))
data3 = np.array(data3['data'])

data4 = arff.load(open('./data/'+cadena+'_4.arff', 'r'))
data4 = np.array(data4['data'])

data5 = arff.load(open('./data/'+cadena+'_5.arff', 'r'))
data5 = np.array(data5['data'])

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
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = np.concatenate((X1, X2, X3, X4, X5), axis=0)

X = scaler.fit_transform(X)

X1 = X[:X1.shape[0]]
X2 = X[X1.shape[0]:X1.shape[0] + X2.shape[0]]
X3 = X[X1.shape[0] + X2.shape[0]:X1.shape[0] + X2.shape[0] + X3.shape[0]]
X4 = X[X1.shape[0] + X2.shape[0] + X3.shape[0]:X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0]]
X5 = X[X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0]:]

# Realizar la validación cruzada
tasa_class_media = 0
tasa_red_media = 0
evaluacion_media = 0
evaluacion_test_media = 0
accuracy_media = 0
tiempo_medio = 0
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

    if model_type == 'KNN':
        model = modelos.KNN(k)
    elif model_type == 'Relief':
        model = modelos.Relief()
    elif model_type == 'BL':
        model = modelos.BL(np.random.randint(0, 1000))

    model.fit(X_train, y_train)

    # Evaluar el modelo

    tasa_red = model.red_rate()
    tasa_red_media += tasa_red

    tasa_clas = model.clas_rate()
    tasa_class_media += tasa_clas

    evaluacion = model.fitness(clasRate=tasa_clas, redRate=tasa_red)
    evaluacion_media += evaluacion

    accuracy = model.accuracy(X_test, y_test)
    accuracy_media += accuracy

    evaluacion_test = funciones.evaluationFunction(accuracy, tasa_red)
    evaluacion_test_media += evaluacion_test

    time_end = time.time()
    tiempo_medio += time_end - time_start

    print()

    print('----------------------Partición', i+1,'----------------------')

    print('Modelo:', model_type)

    print('Conjunto de datos:', cadena)

    if model_type == 'BL':
        print('Semilla:', model.seed)

    print('Tasa de clasificación:', tasa_clas)

    print('Tasa de reducción:', tasa_red)

    print('Fitness train:', evaluacion)

    print('Fitness test:', evaluacion_test)

    print('Accuracy:', accuracy)

    print('Tiempo', time_end - time_start)

    formatted_output = ','.join(map(str, model.weights))

    print('Pesos:', formatted_output)

    if model_type == 'Relief' or model_type == 'BL':
        features_importantes = np.array(feature_names)[model.features]
        # Seleccionar solo las tres primeras características importantes
        primeras_tres_features_importantes = features_importantes[:3]
        # Seleccionar solo las tres últimas características importantes
        ultimas_tres_features_importantes = features_importantes[-3:]

        print('Características más importantes [de mas importante a menos]: ', primeras_tres_features_importantes)
        print('Características menos importantes [de mas importante a menos]: ', ultimas_tres_features_importantes)
            

# Calcular las medias
tasa_class_media /= 5
tasa_red_media /= 5
evaluacion_media /= 5
evaluacion_test_media /= 5
accuracy_media /= 5
tiempo_medio /= 5

time_total_end = time.time()

print()

print('----------------------Resultados Finales----------------------')

print('Modelo:', model_type)

print('Conjunto de datos:', cadena)

print('Media de la tasa de clasificación:', tasa_class_media)

print('Media de la tasa de reducción:', tasa_red_media)

print('Media del fitness train:', evaluacion_media)

print('Media del fitness test:', evaluacion_test_media)

print('Media del accuracy:', accuracy_media)

print('Media del tiempo:', tiempo_medio)

print()

print('Tiempo total:', time_total_end - time_total_start)

