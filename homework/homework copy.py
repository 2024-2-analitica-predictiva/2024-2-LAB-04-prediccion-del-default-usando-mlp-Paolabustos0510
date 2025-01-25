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

import pandas as pd  #  type: ignore
import numpy as np

# Cargar los datos
train_data = pd.read_csv(
    "files/input/train_default_of_credit_card_clients.csv")
#train_data.head()

test_data = pd.read_csv(
    "files/input/test_default_of_credit_card_clients.csv")
#test_data.head()

# Función para limpiar los datasets
def clean_dataset(data):
    # Renombrar la columna 'default payment next month' a 'default'
    data = data.rename(columns={"default payment next month": "default"})
    # Eliminar la columna 'ID'
    if "ID" in data.columns:
        data = data.drop(columns=["ID"])
    
    # Convertir valores no válidos a NaN
    data['EDUCATION'] = data['EDUCATION'].apply(lambda x: x if x > 0 else np.nan)
    data['MARRIAGE'] = data['MARRIAGE'].apply(lambda x: x if x > 0 else np.nan)

    # Agrupar valores mayores a 4 en la columna 'EDUCATION' como "others" (valor 4)
    data['EDUCATION'] = data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    # Validación explícita: Imprimir valores únicos
    print("Valores únicos en 'EDUCATION' después de agrupar mayores a 4:")
    print(data['EDUCATION'].unique())
    
    # Eliminar registros con valores faltantes
    data = data.dropna()

    # Validación final: Asegurarnos de que no haya valores mayores a 4
    if (data['EDUCATION'] > 4).any():
        print("Error: Existen valores mayores a 4 en 'EDUCATION'")
    else:
        print("Limpieza de 'EDUCATION' completada correctamente.")
    
    return data

# Limpiar los datasets
train_data = clean_dataset(train_data)
test_data = clean_dataset(test_data)

# Confirmar limpieza completa
print("Valores únicos en 'EDUCATION' después de la limpieza final para train_data:")
print(train_data['EDUCATION'].unique())
print("Valores únicos en 'EDUCATION' después de la limpieza final para test_data:")
print(test_data['EDUCATION'].unique())


#train_data.head()
#test_data.head()

# Crear copias de los datasets originales para trabajar
train_data_copy = train_data.copy()
test_data_copy = test_data.copy()

# Dividir los datos en X (características) e y (variable objetivo)
# Para el dataset de entrenamiento
X_train = train_data_copy.drop(columns=["default"])  # Todas las columnas excepto la columna objetivo
y_train = train_data_copy["default"]  # Solo la columna objetivo, que es default

# Para el dataset de prueba
X_test = test_data_copy.drop(columns=["default"])  # Todas las columnas excepto la columna objetivo
y_test = test_data_copy["default"]  # Solo la columna objetivo, que es default

print("Datos divididos:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Verificar valores faltantes en X_train y X_test
print("Valores faltantes en X_train:")
print(X_train.isnull().sum())
print("Valores faltantes en X_test:")
print(X_test.isnull().sum())

print(X_train.dtypes)
print(X_test.dtypes)

# Revisar los valores de las Columnas categóricas
categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]

# Revisar valores únicos en X_train
print("Valores únicos en X_train:")
for col in categorical_columns:
    print(f"{col}: {X_train[col].unique()}")

# Revisar valores únicos en X_test
print("\nValores únicos en X_test:")
for col in categorical_columns:
    print(f"{col}: {X_test[col].unique()}")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

# Definir las columnas categóricas
categorical = ["SEX", "EDUCATION", "MARRIAGE"]
numeric = [col for col in X_train.columns if col not in categorical]

# Preprocesador para las columnas categóricas
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


# Preprocesador para las columnas numéricas
numeric_transformer = StandardScaler()
#numeric_transformer = MinMaxScaler()

# Crear un transformador compuesto
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical),
        ("num", numeric_transformer, numeric)
    ]
)

# Crear el pipeline que incluye todo el procesamiento y el modelo MLP
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),                         # Preprocesamiento
    ('selectkbest', SelectKBest(score_func=f_classif)),
    ('pca', PCA()),                                         # Reducción de dimensionalidad con PCA
    ('mlp', MLPClassifier()), # Selección de las k mejores características
    #('mlp', MLPClassifier(max_iter=200, random_state=42)), DAVID
    #('mlp', MLPClassifier(max_iter=14000, random_state=42))                                # Modelo de red neuronal MLP
])

# Ahora el pipeline está listo para ajustarse a los datos
# Se puede hacer grid search o ajustar el modelo aquí. Se hará con la primera opción.

print("Pipeline creado con éxito:")

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Definir los parámetros a optimizar
param_grid = {
    'pca__n_components': [None],
    'selectkbest__k':[20],
    "mlp__random_state":[43],
    "mlp__hidden_layer_sizes": [(51, 31, 41)],
    'mlp__alpha': [0.26],
    "mlp__learning_rate_init": [0.001],
}

# param_grid = {
#     'pca__n_components':[18],
#     'selectkbest__k':[18],
#     'mlp__hidden_layer_sizes': [(142,22)],
#     'mlp__activation': ['logistic'], 
#     'mlp__solver': ['adam'],   
#     'mlp__alpha': [0.0001],  
#     'mlp__learning_rate': ['constant'], 
# }


# param_grid = {
#     'pca__n_components': [17,18],  # Varias opciones para la varianza explicada
#     'selectkbest__k': [20],              # Varias opciones para el número de características seleccionadas
#     'mlp__hidden_layer_sizes': [(50, 30, 40, 60)],  # Varias opciones para el tamaño de las capas ocultas
#     'mlp__alpha': [0.25 , 0.26],        # Varias opciones para el parámetro de regularización
#     'mlp__learning_rate_init': [0.001, 0.0001],  # Varias opciones para la tasa de aprendizaje inicial
#     'mlp__activation': ['relu'],
#     'mlp__solver': ['adam'],
# }

# Crear un scorer de precisión balanceada
#scorer = make_scorer(balanced_accuracy_score)

# Configurar el GridSearchCV para optimizar los hiperparámetros
grid_search = GridSearchCV(
    estimator=pipeline,                 # El pipeline que definimos anteriormente
    param_grid=param_grid,              # El diccionario de parámetros a optimizar
    cv=10,                              # Validación cruzada con 10 splits
    scoring='balanced_accuracy',                     # Usar la métrica de precisión balanceada
    n_jobs=-1,                          # Utilizar todos los núcleos disponibles para paralelizar
    verbose=1,                          # Mostrar el progreso durante la búsqueda                                                      
)
# Ajustar el modelo a los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Imprimir los resultados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)
print(f"Mejor puntuación de validación: {grid_search.best_score_:.4f}")

# Actualizar el pipeline con los mejores parámetros encontrados
best_pipeline = grid_search.best_estimator_



import os
import pickle
import gzip

# Definir la ruta del directorio y archivo
dir_path = '../files/models'
model_path = '../files/models/model.pkl.gz'

# Crear el directorio si no existe
os.makedirs(dir_path, exist_ok=True)

# Guardar el objeto `grid_search` comprimido con gzip
with gzip.open(model_path, 'wb') as f:
    pickle.dump(grid_search, f)

print(f"Modelo guardado exitosamente en {model_path}")

import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import os

# Asegurarnos de usar el mejor modelo encontrado
final_model = grid_search.best_estimator_

# Realizar predicciones
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# Calcular métricas para el conjunto de entrenamiento
train_metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}

# Calcular métricas para el conjunto de prueba
test_metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

# Definir la ruta del archivo de salida
output_dir = "../files/output"
output_path = os.path.join(output_dir, "metrics.json")

# Crear las carpetas necesarias si no existen
os.makedirs(output_dir, exist_ok=True)

# Guardar las métricas en un archivo JSON
with open(output_path, 'w') as f:
    json.dump(train_metrics, f)
    f.write("\n")
    json.dump(test_metrics, f)

print(f"Métricas guardadas exitosamente en {output_path}")

from sklearn.metrics import confusion_matrix
import json
import os

# Calcular las matrices de confusión
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Crear los diccionarios en el formato solicitado
train_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {
        "predicted_0": int(train_cm[0, 0]),
        "predicted_1": int(train_cm[0, 1])
    },
    'true_1': {
        "predicted_0": int(train_cm[1, 0]),
        "predicted_1": int(train_cm[1, 1])
    }
}

test_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {
        "predicted_0": int(test_cm[0, 0]),
        "predicted_1": int(test_cm[0, 1])
    },
    'true_1': {
        "predicted_0": int(test_cm[1, 0]),
        "predicted_1": int(test_cm[1, 1])
    }
}

# Ruta del archivo metrics.json
output_path = "../files/output/metrics.json"

# Leer el archivo existente y agregar los nuevos datos
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        existing_metrics = [json.loads(line) for line in f]
else:
    existing_metrics = []

# Agregar las matrices de confusión
existing_metrics.append(train_cm_dict)
existing_metrics.append(test_cm_dict)

# Guardar nuevamente el archivo JSON con las métricas actualizadas
with open(output_path, 'w') as f:
    for entry in existing_metrics:
        json.dump(entry, f)
        f.write("\n")

print(f"Matrices de confusión agregadas exitosamente a {output_path}")