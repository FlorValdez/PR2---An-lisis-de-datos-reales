#Funciones ETL del proyecto

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Cargar los datos desde los archivos proporcionados
train_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1tdZHFY3JzNXUg3PZO6A7txKFhMyNvchqxEi72W-2X0k/gviz/tq?tqx=out:csv&sheet=Sheet1')
test_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1jjKaIwOcf5ZAEnt1zkDUX3ORjMhInk__PAsPfxBnLFI/gviz/tq?tqx=out:csv&sheet=Sheet1')

# Convertir la columna 'date' a datetime y extraer características numéricas si es necesario
train_data['date'] = pd.to_datetime(train_data['date'])
# Ejemplo: extraer el mes como una característica numérica
train_data['month'] = train_data['date'].dt.month

# Preprocesamiento de Datos
# Imputar valores nulos con la media para columnas numéricas solamente
numeric_columns_train = train_data.select_dtypes(include=np.number).columns
train_data[numeric_columns_train] = train_data[numeric_columns_train].fillna(train_data[numeric_columns_train].mean())

numeric_columns_test = test_data.select_dtypes(include=np.number).columns
test_data[numeric_columns_test] = test_data[numeric_columns_test].fillna(test_data[numeric_columns_test].mean())

# Separar características y variable objetivo
X_train = train_data[['humidity', 'wind_speed', 'meanpressure']]
y_train = train_data['meantemp']

X_test = test_data[['humidity', 'wind_speed', 'meanpressure']]
y_test = test_data['meantemp']

# Estandarización de las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear nuevas características: Interacciones, polinomios de segundo grado, etc.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Actualizar las características a utilizar
X_train = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out())
X_test = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out())
