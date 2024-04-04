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

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("scikit-learn no está instalado. Instalando scikit-learn...")
    try:
        subprocess.check_call(["pip", "install", "scikit-learn"])
        from sklearn.preprocessing import MinMaxScaler
        print("scikit-learn se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar scikit-learn. Por favor, instálalo manualmente.")

import time
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

print('Elige el modelo a utilizar [Opciones: KNN[DEFAULT][1], Relief[2], BL[3], ALL[4]):')
model_type = input()

if model_type == 'KNN' or model_type == '' or model_type == '1':
    model_type = 'KNN'
elif model_type == 'Relief' or model_type == '2':
    model_type = 'Relief'
elif model_type == 'BL' or model_type == '3':
    model_type = 'BL'
elif model_type == 'ALL' or model_type == '4':
    model_type = 'ALL'
elif model_type != '':
    print('Modelo no válido')
    exit()

k = 1
seed = 7
if(model_type == 'KNN' or model_type == 'ALL'):
    print('Introduce el valor de k [DEFAULT=1]:')
    k = input()
    if k == '':
        k = 1
    elif k != '':
        k = int(k)
    
    if k < 1:
        print('Valor de k no válido')
        exit()

if model_type == 'BL' or model_type == 'ALL':
    print('Introduce el valor de la semilla [DEFAULT=7]:')
    seed_i = input()
    if seed_i == '':
        seed = 7
    else:
        seed = int(seed_i)

print('Guardar resultados en archivo CSV [S/N][DEFAULT=N]:')
guardar = input().lower()

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

scaler = MinMaxScaler()

X = np.concatenate((X1, X2, X3, X4, X5), axis=0)

X = scaler.fit_transform(X)

X1 = X[:X1.shape[0]]
X2 = X[X1.shape[0]:X1.shape[0] + X2.shape[0]]
X3 = X[X1.shape[0] + X2.shape[0]:X1.shape[0] + X2.shape[0] + X3.shape[0]]
X4 = X[X1.shape[0] + X2.shape[0] + X3.shape[0]:X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0]]
X5 = X[X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0]:]

# 5-fold cross-validation

if model_type == 'ALL':
    models = ['KNN', 'Relief', 'BL']
else:
    models = [model_type]

for m in models:
    print('Dataset:', cadena)
    print('Modelo:', m)
    if m == 'KNN':
        print('k:', k)
    elif m == 'BL':
        print('Semilla:', seed)

    df, pesos = funciones.fiveCrossValidation(X1, X2, X3, X4, X5, y1, y2, y3, y4, y5, m, seed, k)

    print('Pesos:')
    print(pesos)

    print()

    print('Resultados:')
    print(df)

    print()

    print('Estadísticas:')

    print(df.describe().loc[['mean', 'std', 'min', 'max']])

    if guardar == 's':
        print()
        print('Guardando resultados en archivo CSV...')
        df.to_csv('./results/'+cadena+'_'+m+'.csv', index=False)
        print('Resultados guardados correctamente')
        print()



time_total_end = time.time()

print()

print('Tiempo total:', time_total_end - time_total_start)

