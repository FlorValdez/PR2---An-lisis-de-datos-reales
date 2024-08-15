#Funciones ML del proyecto
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
