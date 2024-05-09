# Manual de Usuario del Programa de Selección de Base de Datos y Modelo

Este programa te permite seleccionar una base de datos y un modelo para su análisis. A continuación, se presenta un manual detallado para su uso:

## Inicio del Programa

Para ejecutar el script, usar `python3 main.py`.

Al ejecutar el script, se te solicitará introducir la base de datos que deseas utilizar. Las opciones disponibles son:

- **BreastCancer** (Por defecto): Base de datos sobre cáncer de mama.
- **Ecoli**: Base de datos sobre secuencias de ADN de bacterias E. coli.
- **Parkinson**: Base de datos sobre la detección de la enfermedad de Parkinson.

## Selección del Modelo

Después de seleccionar la base de datos, se te pedirá que elijas el modelo de aprendizaje automático que deseas utilizar. Las opciones son:

- **KNN** (Por defecto): K-Nearest Neighbors.
- **Relief**: Algoritmo de selección de características Relief.
- **BL**: Algoritmo Búsqueda Local.
- **AGG**: Algoritmo Genético Generacional.
- **AGE**: Algoritmo Genético Estacionario.
- **AM**: Algoritmo Memético.
- **ALL**: Ejecutar todos los modelos disponibles.

## Configuración Adicional

Dependiendo del modelo seleccionado, puede ser necesario introducir configuraciones adicionales:

- **Valor de k**: Relevante para el modelo KNN y la opción ALL. Representa el número de vecinos más cercanos a considerar.
- **Valor de la semilla**: Requerido para los modelos BL, AGG, AGE, AM y la opción ALL. La semilla se utiliza para inicializar la generación de números aleatorios.
- **Versión Mejorada**: Para los modelos AGG, AGE, AM y la opción ALL, se te preguntará si deseas utilizar la versión mejorada de los algoritmos genéticos.
- **Operador de Cruce**: Para AGG y AGE, se te solicitará seleccionar un operador de cruce entre 'CA' (Cruce Aritmético) y 'BLX' (Blend Crossover).
- **Operador de Selección en BL para el algoritmo memético**: Para el modelo AM, deberás seleccionar un operador de selección entre 'All', 'Random' y 'Best'.

## Ejecución del Modelo

Una vez que se han introducido todas las configuraciones, el modelo seleccionado se ejecutará con las opciones especificadas.

## Guardar Resultados en Archivo CSV

Puedes seleccionar `S` para guardar los resultados en un archivo CSV o `N` para no guardarlos.

## Requisitos del Programa

Este programa requiere los siguientes paquetes de Python instalados en tu sistema:

- numpy
- scipy
- pandas
- liac-arff
- scikit-learn
- tqdm

## Errores y Salida del Programa

Si introduces una base de datos o modelo no válido, el programa mostrará un mensaje de error correspondiente y se cerrará.

