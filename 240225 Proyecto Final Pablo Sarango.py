""" 
- Variable asignada:    espacio_lavado     
- Población objetivo:   region == "Sierra" 

"""
# --------------- IMPORTACIÓN DE LIBRERIAS NECESARIAS  --------------- #

# Importamos numpy para realizar operaciones numéricas eficientes.
import numpy as np

# Pandas nos permitirá trabajar con conjuntos de datos estructurados.
import pandas as pd

# Desde sklearn.model_selection importaremos funciones para dividir conjuntos de datos y realizar validación cruzada.
from sklearn.model_selection import train_test_split, KFold

# Utilizaremos sklearn.preprocessing para preprocesar nuestros datos antes de entrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler

# sklearn.metrics nos proporcionará métricas para evaluar el rendimiento de nuestros modelos.
from sklearn.metrics import accuracy_score

# statsmodels.api nos permitirá realizar análisis estadísticos más detallados y estimación de modelos.
import statsmodels.api as sm

# Por último, matplotlib.pyplot nos ayudará a visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt

# --------------- IMPORTACION Y LIMPIEZA DE DATOS  --------------- #

# Impotar los datos originales
datos = pd.read_csv("data/txt/sample_endi_model_10p.txt", sep = ";")    

# Convertimos los códigos numéricos de las regiones en etiquetas más comprensibles
datos["region"] = datos["region"].apply(
    lambda x: 
    "Costa" if x == 1 
    else "Sierra" 
    if x == 2 
    else "Oriente")

# Filtramos el df según la población objetivo 
datos = datos[datos['region'] == 'Sierra']

# Selecionamos SOLO las filas no contengan valores NA 
datos = datos[~datos["dcronica"].isna()]

datos.isna().sum()

# Seleccionamos las variables de interes 
variables = ['n_hijos', 'sexo', 'condicion_empleo', 'espacio_lavado', 'region']

# Eliminamos las filas con valores nulos en cada variable
for i in variables:
    datos = datos[~datos[i].isna()]

# Visualizamos el dataframe limpio 
datos.info

""" Ejercicio 1: Exploración de Datos

1) Calcular ¿cuántos niños se encuentran en la region sierra 
2) Calcular el numero de niños que viven en region sierra que cuentan con un espacio de lavado de manos

"""
# Calcular el numero de niños que se encuentran en la region sierra (poblacion objetivo)
num_ninos_poblacion_objetivo = (datos["region"] == "Sierra").sum()

# Calcular el conteo de niños que viven en region sierra que cuentan con un espacio de lavado de manos (variable asignada)
count_espacio_lavado = (datos["espacio_lavado"] == 1).sum()

print("El numero de niños que se encuentran en la region sierra (poblacion objetivo) es:", num_ninos_poblacion_objetivo)
print("El numero de niños qque viven en region sierra que cuentan con un espacio de lavado de manos (variable asignada) es:", count_espacio_lavado)

# --------------- TRANSFORMACIONES DE VARIABLES --------------- #

# Definimos las variables categóricas y numéricas que utilizaremos en nuestro análisis
variables_categoricas = ['region', 'sexo', 'condicion_empleo', 'espacio_lavado']
variables_numericas = ['n_hijos']

# Creamos un transformador para estandarizar las variables numéricas y una copia de nuestros datos para no modificar el conjunto original
transformador = StandardScaler()
datos_escalados = datos.copy()

# Estandarizamos las variables numéricas utilizando el transformador
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

# Convertimos las variables categóricas en variables dummy utilizando one-hot encoding
datos_dummies = pd.get_dummies(datos_escalados, drop_first = True)

datos_dummies.info()

# Seleccionamos las variables predictoras (X) y la variable ohttps://www.youtube.com/watch?v=Hr-K0Eke5KU&pp=ygUP7Jyk7IOBICBsb3ZlIGlzbjetivo (y) para nuestro modelo
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años', 'espacio_lavado']]
y = datos_dummies["dcronica"]

# Definimos los pesos asociados a cada observación para considerar el diseño muestral
weights = datos_dummies['fexp_nino']

# --------------- SEPARACIÓN DE LAS MUESTRAS EN ENTRAMIENTO (TRAIN) Y PRUBEA (TEST) --------------- #

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

print(X_train.isna().sum())
X_train.dtypes


""" Ejercicio 2: Modelo Logit

1) ¿Cuál es el valor del parámetro asociado a la variable clave si ejecutamos el modelo 
solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento? ¿Es significativo?

El coeficiente estimado para la variable asignada es -0.1.6517, 
lo que indica que hay una asociación negativa entre la variable asignada y la desnutricion cronica.

El valor p (P>|z|) asociado a este coeficiente es menor que 0.05, 
lo que indica que hay una diferencia estadísticamente significativa 
entre la presencia y ausencia de un espacio de lavado en relación con la variable dependiente "dcronica".
"""

# --------------- AJUSTE DEL MODELO PROBANDO CON TEST DATA --------------- #

modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)

# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)

# Comparamos las predicciones con los valores reales
predictions_class == y_test
print("La precisión promedio del modelo testeando con datos test es", np.mean(predictions_class))
 
# --------------- AJUSTE DEL MODELO PROBANDO CON TRAIN DATA --------------- #

# Realizamos predicciones en el conjunto de entrenamiento
predictions_train = result.predict(X_train)

# Convertimos las probabilidades en clases binarias
predictions_train_class = (predictions_train > 0.5).astype(int)

# Comparamos las predicciones con los valores reales
predictions_train_class == y_train
print("La precisión promedio del modelo testeando con datos train es", np.mean(predictions_train_class))


# --------------- VALIDACIÓN CRUZADA --------------- #

# 100 folds:
kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

# --------------- VALIDACIÓN CRUZADA: PRECISIÓN DEL MODELO --------------- #

# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

# --------------- VALIDACIÓN CRUZADA: EL COMPORTAMIENTO DEL PARAMETRO ASOCIADO A N_HIJOS --------------- #

plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["n_hijos"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["n_hijos"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["n_hijos"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

""" Ejercicio 3: Evaluación del Modelo con Datos Filtrados

1) ¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado? 

- Precisión promedio de validación cruzada para: 
Modelo sin flitrar (Obtenido de la guía) = 0.731372549019608
Modelo filtrado (Incluyendo la variable y poblacion objetivo) = 0.8398907103825134

Al incluir la variable "espacio_lavado" la precisión promedio aumenta un 11%

2) ¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior?

- Media de coeficientes para: 
Modelo sin flitrar (Obtenido de la guía) = 0.11
Modelo filtrado (Incluyendo la variable y poblacion objetivo) = 0.09

Al incluir la variable "espacio_lavado" la media del coeficiente disminuye un 2%


"""