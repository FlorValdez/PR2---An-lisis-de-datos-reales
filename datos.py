import os
import pandas as pd
import requests

# Crear las carpetas si no existen
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# URLs de los datos
urls = [
    'https://docs.google.com/spreadsheets/d/1tdZHFY3JzNXUg3PZO6A7txKFhMyNvchqxEi72W-2X0k/gviz/tq?tqx=out:csv&sheet=Sheet1',
    'https://docs.google.com/spreadsheets/d/1jjKaIwOcf5ZAEnt1zkDUX3ORjMhInk__PAsPfxBnLFI/gviz/tq?tqx=out:csv&sheet=Sheet1'
]

# Nombres de los archivos que se guardarán en la carpeta raw
filenames = [
    'data/raw/datos_brutos_1.csv',
    'data/raw/datos_brutos_2.csv'
]

# Descargar y guardar los archivos
for url, filename in zip(urls, filenames):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

print("Datos brutos descargados y guardados en la carpeta 'data/raw'")

# Mostrar los datos brutos
data1 = pd.read_csv('data/raw/datos_brutos_1.csv')
data2 = pd.read_csv('data/raw/datos_brutos_2.csv')

print("\nDatos Brutos 1:")
print(data1.head())  # Muestra las primeras 5 filas del primer archivo

print("\nDatos Brutos 2:")
print(data2.head())  # Muestra las primeras 5 filas del segundo archivo

# Realizar transformaciones

# 1. Concatenar los dos archivos
data_combined = pd.concat([data1, data2])

# 2. Eliminar filas duplicadas
data_combined = data_combined.drop_duplicates()


# 3. Manejar valores faltantes (eliminar filas con valores faltantes)
data_cleaned = data_combined.dropna() # Use data_combined here


# Mostrar los datos transformados antes de guardarlos
print("\nDatos Transformados:")
print(data_cleaned.head())  # Muestra las primeras 5 filas de los datos transformados

# Guardar los datos transformados en la carpeta processed
data_cleaned.to_csv('data/processed/datos_transformados.csv', index=False)

print("\nDatos transformados guardados en la carpeta 'data/processed'")