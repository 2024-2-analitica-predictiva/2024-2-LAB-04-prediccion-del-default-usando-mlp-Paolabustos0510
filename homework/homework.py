# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
from sklearn.metrics import precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import json
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
import gzip
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import decomposition
import numpy as np


#agrupar educaciones mayores que 4 (Others)
#def agrupar_educaciones(codigo_educacion):
    #if codigo_educacion > 4:
        #return 4
    #return codigo_educacion


#def cargar_limpiar_dataset(nombre_archivo):
    #datos = pd.read_csv(nombre_archivo)
    
    # - Renombre la columna "default payment next month" a "default".
    #datos = datos.rename(columns={"default payment next month" : "default"})

    # Remueva la columna "ID".
    #datos.drop(columns=["ID"], inplace = True)

    # Elimine los registros con informacion no disponible.
    #datos = datos.dropna()
    #datos = datos[(datos["EDUCATION"]!=0) & (datos['MARRIAGE']!=0)]

    # Para la columna EDUCATION, valores > 4 indican niveles superiores de educación, agrupe estos valores en la categoría "others"
    #datos["EDUCATION"] = datos["EDUCATION"].apply(agrupar_educaciones)

    #return datos

# Función para limpiar los datasets
def cargar_limpiar_dataset(nombre_archivo):
    datos = pd.read_csv(nombre_archivo)

    # Renombrar la columna 'default payment next month' a 'default'
    datos = datos.rename(columns={"default payment next month": "default"})
    # Eliminar la columna 'ID'
    if "ID" in datos.columns:
        datos = datos.drop(columns=["ID"])
    
    # Convertir valores no válidos a NaN
    datos['EDUCATION'] = datos['EDUCATION'].apply(lambda x: x if x > 0 else np.nan)
    datos['MARRIAGE'] = datos['MARRIAGE'].apply(lambda x: x if x > 0 else np.nan)

    # Agrupar valores mayores a 4 en la columna 'EDUCATION' como "others" (valor 4)
    datos['EDUCATION'] = datos['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    # Validación explícita: Imprimir valores únicos
    print("Valores únicos en 'EDUCATION' después de agrupar mayores a 4:")
    print(datos['EDUCATION'].unique())
    
    # Eliminar registros con valores faltantes
    datos = datos.dropna()

    # Validación final: Asegurarnos de que no haya valores mayores a 4
    if (datos['EDUCATION'] > 4).any():
        print("Error: Existen valores mayores a 4 en 'EDUCATION'")
    else:
        print("Limpieza de 'EDUCATION' completada correctamente.")
    
    return datos 

datos_entrenamiento = cargar_limpiar_dataset("files/input/train_default_of_credit_card_clients.csv")
datos_prueba = cargar_limpiar_dataset("files/input/test_default_of_credit_card_clients.csv")

print(datos_prueba.head())

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train = datos_entrenamiento.drop(columns=["default"])
y_train = datos_entrenamiento["default"]

x_test = datos_prueba.drop(columns=["default"])
y_test = datos_prueba["default"]

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.

# Transformación variables categóricas usando el método one-hot-encoding.
variables_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
variables_numericas = [col for col in x_train.columns if col not in variables_categoricas]

transformer_variables_categoricas = OneHotEncoder()
transformer_variables_numericas = "passthrough"

preprocesador = ColumnTransformer(
    transformers=[
        ("categoricas", transformer_variables_categoricas, variables_categoricas),
        ("numericas", transformer_variables_numericas, variables_numericas)
    ],
)

#Análisis PCA
pca = PCA()

#Estandarización
estandarizador = StandardScaler()

# Elección Mejores K
mejores_k = SelectKBest(score_func=f_classif)

 #Crear modelo de Red Neuronal tipo MLP.
modelo = MLPClassifier()

# Pipeline completo
pipeline = Pipeline(
    steps=[
        ("preprocesador", preprocesador), 
        ("pca", pca), 
        ("estandarizador", estandarizador),
        ("mejoresK", mejores_k),
        ("MLP", modelo),     
    ]
)

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

# Establecer hiperparámetros a evaluar
param_grid = [
    {
        'MLP__hidden_layer_sizes'     : [(51, 31, 41)],
        'pca__n_components'           : [None],
	    'MLP__learning_rate_init'     : [0.001],
        'mejoresK__k'                 : [20],
        'MLP__alpha'                  : [0.26],
        "MLP__random_state"           : [43]
    #     'activation'             : ['relu', ],
    } 
]

# Creación malla de hiperpárametros
busqueda_malla = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=10,
    verbose=1,
    n_jobs=-1,
)

# Entrenamiento de modelo
busqueda_malla.fit(x_train, y_train)



#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

mejor_modelo = busqueda_malla.best_estimator_
mejores_parametros = busqueda_malla.best_params_
mejor_resultado = busqueda_malla.best_score_
print("Mejor resultado: ", mejor_resultado)

with gzip.open("files/models/model.pkl.gz", "w") as archivo:
    pickle.dump(busqueda_malla, archivo)

#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#

def calcular_metricas(modelo, x, y, dataset): 
    y_pred = modelo.predict(x) 
    diccionario_metricas = {
        "type" : "metrics",
        "dataset" : dataset,
        "precision" : precision_score(y, y_pred),
        "balanced_accuracy" : balanced_accuracy_score(y, y_pred), 
        "recall" : recall_score(y, y_pred), 
        "f1_score" : f1_score(y, y_pred), 
    }

    return diccionario_metricas

# Función para calcular matriz de confusión
def calcular_matriz_confusion(modelo, x, y, dataset):
    matriz_con = confusion_matrix(y, modelo.predict(x))
    diccionario_matriz = {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {"predicted_0": int(matriz_con[0, 0]), "predicted_1": int(matriz_con[0, 1])},
        "true_1": {"predicted_0": int(matriz_con[1, 0]), "predicted_1": int(matriz_con[1, 1])},
    }
    return diccionario_matriz

# Guardar métricas y matrices de consufión
valores = [
    calcular_metricas(mejor_modelo, x_train, y_train, "train"),
    calcular_metricas(mejor_modelo, x_test, y_test, "test"),
    calcular_matriz_confusion(mejor_modelo, x_train, y_train,"train"),
    calcular_matriz_confusion(mejor_modelo, x_test, y_test, "test"),
]

# Revisar valores
print(valores)

# Guardar archivo JSON
with open("files/output/metrics.json", "w") as archivo:
    for v in valores:
        json.dump(v, archivo)
        archivo.write("\n")


# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
