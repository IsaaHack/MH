# Manual de Usuario del Programa de Selección de Base de Datos y Modelo

Este programa te permite seleccionar una base de datos y un modelo para su análisis. A continuación, se presenta un manual detallado para su uso:

## Inicio del Programa:
- Al iniciar el programa, verás el siguiente mensaje:
```
Introduce la base de datos a utilizar [Opciones: BreastCancer[DEFAULT][1], Ecoli[2], Parkinson[3]]:
```

- Debes introducir el nombre de la base de datos que deseas utilizar. Las opciones son `BreastCancer`, `Ecoli` y `Parkinson`. También puedes seleccionar simplemente ingresando el número correspondiente a la base de datos deseada.

## Selección del Modelo:
- Después de seleccionar la base de datos, el programa te pedirá que elijas el modelo a utilizar con el siguiente mensaje:
```
Elige el modelo a utilizar [Opciones: KNN[DEFAULT][1], Relief[2], BL[3], ALL[4]]:
```
- Puedes seleccionar el modelo escribiendo su nombre completo o simplemente ingresando el número correspondiente al modelo deseado. Las opciones disponibles son `KNN`, `Relief`, `BL` y `ALL`. Si eliges `ALL`, se utilizarán todos los modelos disponibles.

## Configuración Adicional según el Modelo:
- Dependiendo del modelo seleccionado, es posible que se soliciten configuraciones adicionales:
- **Para KNN:**
  - Se te pedirá que introduzcas el valor de `k`, que representa el número de vecinos a considerar.
- **Para BL:**
  - Se te pedirá que introduzcas el valor de la semilla que se utilizará en el proceso.

## Guardar Resultados en Archivo CSV:
- Después de seleccionar el modelo y configurar las opciones adicionales (si es necesario), se te preguntará si deseas guardar los resultados en un archivo CSV con el siguiente mensaje:
```
Guardar resultados en archivo CSV [S/N][DEFAULT=N]:
```
- Puedes seleccionar `S` para guardar los resultados en un archivo CSV o `N` para no guardarlos.

## Requisitos del Programa:
- Este programa requiere los siguientes paquetes de Python instalados en tu sistema:
- `numpy`
- `scipy`
- `pandas`
- `liac-arff`
- `scikit-learn`

## Errores y Salida del Programa:
- Si introduces una base de datos o modelo no válido, el programa mostrará un mensaje de error correspondiente y se cerrará.
