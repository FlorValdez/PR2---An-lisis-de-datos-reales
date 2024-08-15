#PR2 - Análisis de datos reales

#1. Plan de Solución
# Requisitos del Proyecto:
# Lenguaje: Python.
# Base de datos.
# Entorno de Desarrollo: Visual Studio Code, Jupyter Notebook.
#Librerías: pandas, numpy, scikit-learn, matplotlib, seaborn.

#2. Objetivos:
# Desarrollar un modelo de regresión que prediga la temperatura promedio diaria basada en la humedad, velocidad del viento y presión atmosférica.
# Evaluar la precisión del modelo utilizando métricas como el error cuadrático medio (MSE) y la R².

#3. Pasos Clave:
#Exploración de Datos: Entender la distribución de las variables y las relaciones entre ellas.
#Preprocesamiento: Manejo de valores nulos, normalización de variables si es necesario.
#Selección de Modelo: Probar diferentes algoritmos de regresión (p. ej., regresión lineal, árbol de decisión).
#Entrenamiento y Evaluación: Entrenar el modelo con los datos de entrenamiento y evaluarlo con los datos de prueba.
#Visualización de Resultados: Generar gráficos que ilustren el rendimiento del modelo.

#Configuración del Entorno y Carga de Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos desde los archivos proporcionados
train_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1tdZHFY3JzNXUg3PZO6A7txKFhMyNvchqxEi72W-2X0k/gviz/tq?tqx=out:csv&sheet=Sheet1')
test_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1jjKaIwOcf5ZAEnt1zkDUX3ORjMhInk__PAsPfxBnLFI/gviz/tq?tqx=out:csv&sheet=Sheet1')

# Convertir la columna 'date' a datetime y extraer características numéricas si es necesario
train_data['date'] = pd.to_datetime(train_data['date'])
# Ejemplo: extraer el mes como una característica numérica
train_data['month'] = train_data['date'].dt.month

# Exploración inicial de los datos
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# Visualización inicial de la distribución de las variables
sns.pairplot(train_data.drop('date', axis=1))
plt.show()

# Correlación entre las variables (excluyendo 'date')
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.drop('date', axis=1).corr(), annot=True, cmap='coolwarm')
plt.show()

# Preprocesamiento de Datos
# Imputar valores nulos con la media para columnas numéricas solamente
numeric_columns_train = train_data.select_dtypes(include=np.number).columns
train_data[numeric_columns_train] = train_data[numeric_columns_train].fillna(train_data[numeric_columns_train].mean())

numeric_columns_test = test_data.select_dtypes(include=np.number).columns
test_data[numeric_columns_test] = test_data[numeric_columns_test].fillna(test_data[numeric_columns_test].mean())

# Estilo y configuración de colores
sns.set(style="whitegrid", palette="pastel")

# Visualización de outliers usando diagramas de caja
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_data[['humidity', 'wind_speed', 'meanpressure', 'meantemp']],
            palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Detección de Outliers', fontsize=16, fontweight='bold')
plt.show()

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


# Análisis Exploratorio de Datos (EDA) Avanzado
# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(10, 6))
sns.histplot(train_data['meantemp'], kde=True)
plt.title('Distribución de la Temperatura Media')
plt.show()


# Correlación entre las variables
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación')
plt.show()

# Convertir fecha a tipo datetime y analizar tendencias
train_data['date'] = pd.to_datetime(train_data['date'])
train_data.set_index('date', inplace=True)

plt.figure(figsize=(12, 6))
train_data['meantemp'].plot()
plt.title('Tendencia Temporal de la Temperatura Media')
plt.show()

# Selección y Optimización del Modelo
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}

results_df = pd.DataFrame(results).T
print(results_df)

# Validación cruzada para un modelo específico
cross_val_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cross_val_mse = -cross_val_scores.mean()
print(f'Cross-Validated MSE: {cross_val_mse}')

# Ejemplo de GridSearchCV para Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f'Mejor modelo: {best_model}')


# Entrenamiento y Evaluación del Modelo
# Entrenar el mejor modelo encontrado
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# Evaluación
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f'MSE del Mejor Modelo: {mse_best}')
print(f'R² del Mejor Modelo: {r2_best}')


# Análisis de Resultados
# Importancia de características para Random Forest o cualquier modelo con feature_importances_
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Importancia de las Características')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
plt.show()

# Análisis de errores
errors = y_test - y_pred_best
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.title('Distribución de Errores')
plt.show()

# Comparar los resultados de diferentes modelos
print(results_df.sort_values(by='MSE'))


# Visualización Avanzada de Resultados
# Gráfico de dispersión de predicciones vs valores reales para el mejor modelo
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Línea de identidad
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')



