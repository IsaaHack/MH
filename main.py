import subprocess
import os

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
    if arff.__package__ == 'arff':
        print("arff no está instalado correctamente. Por favor, instálalo manualmente.")
        print('Desinstala arff con "pip uninstall arff" y vuelve a instalarlo con "pip install liac-arff".')
        
        exit()
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

try:
    from tabulate import tabulate
except ImportError:
    print("tabulate no está instalado. Instalando tabulate...")
    try:
        subprocess.check_call(["pip", "install", "tabulate"])
        import tabulate
        print("tabulate se ha instalado correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar tabulate. Por favor, instálalo manualmente.")

import time
import funciones


def main():
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

    ALL_MODELS = ['KNN', 'Relief', 'BL', 'AGG' 'AGE', 'AM-']
    print('Elige el modelo a utilizar [Opciones: KNN[DEFAULT][1], Relief[2], BL[3], AGG[4], AGE[5], AM[6], ALL[7]]:')
    model_type = input()

    if model_type == 'KNN' or model_type == '' or model_type == '1':
        model_type = 'KNN'
    elif model_type == 'Relief' or model_type == '2':
        model_type = 'Relief'
    elif model_type == 'BL' or model_type == '3':
        model_type = 'BL'
    elif model_type == 'AGG' or model_type == '4':
        model_type = 'AGG'
    elif model_type == 'AGE' or model_type == '5':
        model_type = 'AGE'
    elif model_type == 'AM' or model_type == '6':
        model_type = 'AM'
    elif model_type == 'ALL' or model_type == '7':
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

    if model_type == 'BL' or model_type == 'AGG' or model_type == 'AGE' or model_type == 'AM' or model_type == 'ALL':
        print('Introduce el valor de la semilla [DEFAULT=7]:')
        seed_i = input()
        if seed_i == '':
            seed = 7
        else:
            seed = int(seed_i)

    version_mejorada = False
    if model_type == 'AGG' or model_type == 'AGE' or model_type == 'AM' or model_type == 'ALL':
        print('¿Quieres utilizar la versión mejorada de los algoritmos genéticos? [S/N][DEFAULT=N]:')
        version_mejorada = input().lower() == 's'

    if model_type == 'AGG' or model_type == 'AGE':
        print('Introduce el operador de cruce [Opciones: CA[DEFAULT][1], BLX[2]], ALL[3]:')
        cruce = input()
        if cruce == '' or cruce == '1' or cruce == 'CA':
            cruce = 'CA'
        elif cruce == 'BLX' or cruce == '2':
            cruce = 'BLX'
        elif cruce == 'ALL' or cruce == '3':
            cruce = 'ALL'
        else:
            print('Operador de cruce no válido')
            exit()

    seleccion = ''
    if model_type == 'AM':
        print('Introduce el operador seleción en BL [Opciones: (10,1.0)[DEFAULT][1], (10,0.1)[2], (10,0.1mej)[3]]:')
        seleccion = input()
        if seleccion == '' or seleccion == '1' or seleccion == '(10,1.0)':
            seleccion = 'All'
        elif seleccion == '(10,0.1)' or seleccion == '2':
            seleccion = 'Random'
        elif seleccion == '(10,0.1mej)' or seleccion == '3':
            seleccion = 'Best'
        else:
            print('Operador de selección no válido')
            exit()


    np.random.seed(seed)

    match model_type:
        case 'KNN':
            model = ['KNN']
            model_name = [str(k)+'NN']
            params = [{'k' : k}]
        case 'Relief':
            model = ['Relief']
            model_name = ['Relief']
            params = [{}]
        case 'BL':
            model_name = ['BL']
            model = ['BL']
            params = [{}]
        case 'AGG':
            if cruce == 'ALL':
                model = ['AGG', 'AGG']
                model_name = ['AGG-CA', 'AGG-BLX']
                params = [{'crossover' : 'CA', 'improved' : version_mejorada},
                          {'crossover' : 'BLX', 'improved' : version_mejorada}]
            else:
                model = ['AGG']
                model_name = ['AGG-'+cruce]
                params = [{'crossover' : cruce, 'improved' : version_mejorada}]
        case 'AGE':
            if cruce == 'ALL':
                model = ['AGE', 'AGE']
                model_name = ['AGE-CA', 'AGE-BLX']
                params = [{'crossover' : 'CA', 'improved' : version_mejorada},
                          {'crossover' : 'BLX', 'improved' : version_mejorada}]
            else:
                model = ['AGE']
                model_name = ['AGE-'+cruce]
                params = [{'crossover' : cruce, 'improved' : version_mejorada}]
        case 'AM':
            model = ['AM']
            model_name = ['AM-'+seleccion]
            params = [{'crossover' : 'BLX', 'bl_selection' : seleccion, 'improved' : version_mejorada}]
        case 'ALL':
            model = ['KNN', 'Relief', 'BL', 'AGG', 'AGG', 'AGE', 'AGE', 'AM', 'AM', 'AM']
            model_name = [str(k)+'NN', 'Relief', 'BL', 'AGG-CA', 'AGG-BLX', 'AGE-CA', 'AGE-BLX', 'AM-All', 'AM-Random', 'AM-Best']
            params = [{'k' : k},
                       {}, 
                       {},
                          {'crossover' : 'CA', 'improved' : version_mejorada},
                          {'crossover' : 'BLX', 'improved' : version_mejorada},
                            {'crossover' : 'CA', 'improved' : version_mejorada},
                            {'crossover' : 'BLX', 'improved' : version_mejorada},
                              {'crossover' : 'BLX', 'bl_selection' : 'All', 'improved' : version_mejorada},
                              {'crossover' : 'BLX', 'bl_selection' : 'Random', 'improved' : version_mejorada},
                              {'crossover' : 'BLX', 'bl_selection' : 'Best', 'improved' : version_mejorada}

                        ]
        
    models_params = funciones.Model_Parameters(model, params, model_name)

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

    models = models_params.model_type
    params = models_params.params
    model_names = models_params.model_name

    global_dataframe = pd.DataFrame()

    for m, model_params, model_name in zip(models, params, model_names):
        print()
        print('Dataset:', cadena)
        print('Modelo:', model_name)
        print('Parámetros:', model_params)

        df, pesos = funciones.fiveCrossValidation(X1, X2, X3, X4, X5, y1, y2, y3, y4, y5, m, model_params, seed)

        print('Pesos:')
        print(pesos)

        print()

        print('Resultados:')
        print(tabulate(df, headers='keys', numalign='center', showindex=False))

        print()

        print('Estadísticas:')

        print(tabulate(df.describe().loc[['mean', 'std', 'min', 'max']], headers='keys', numalign='center'))

        print()

        if guardar == 's':
            print()
            print('Guardando resultados en archivo CSV...')
            if not os.path.exists('./results'):
                os.makedirs('./results')
            try:
                if 'improved' in model_params:
                    if model_params['improved']:
                        df.to_csv('./results/'+cadena+'_'+model_name+'_improved.csv', index=False)
                    else:
                        df.to_csv('./results/'+cadena+'_'+model_name+'.csv', index=False)
                else:
                    df.to_csv('./results/'+cadena+'_'+model_name+'.csv', index=False)
                print('Resultados guardados correctamente')
            except:
                print('Error al guardar los resultados')
            print()

        global_dataframe = pd.concat([global_dataframe, df], axis=1)


    if guardar == 's' and model_type == 'ALL':
        print()
        print('Guardando resultados globales en archivo CSV...')
        if not os.path.exists('./results'):
            os.makedirs('./results')
        try:
            if version_mejorada:
                global_dataframe.to_csv('./results/'+cadena+'_ALL_improved.csv', index=False)
            else:
                global_dataframe.to_csv('./results/'+cadena+'_ALL.csv', index=False)
            print('Resultados guardados correctamente')
        except:
            print('Error al guardar los resultados')
        print()


    time_total_end = time.time()

    print()

    print('Tiempo total:', time_total_end - time_total_start)


if __name__ == '__main__':
    main()

