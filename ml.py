#Funciones ML


# Separar características (X) y variable objetivo (y) del conjunto de datos de entrenamiento y prueba
X_train = train_data[['humidity', 'wind_speed', 'meanpressure']]
y_train = train_data['meantemp']

X_test = test_data[['humidity', 'wind_speed', 'meanpressure']]
y_test = test_data['meantemp']
