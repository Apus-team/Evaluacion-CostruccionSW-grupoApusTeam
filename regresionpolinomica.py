import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset (Asegúrate de que el CSV esté en la misma carpeta)
df = pd.read_csv('USA_Housing.csv')

# 2. Seleccionar variables (X en 2D, y en 1D)
X = df[['Avg. Area Income']].values 
y = df['Price'].values

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Aplicar transformación polinómica (Grado 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# 4. Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_poly_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = modelo.predict(X_poly_test)

# Calcular e imprimir métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- RESULTADOS DEL MODELO POLINÓMICO (GRADO 2) ---")
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente R^2: {r2}")
print("--------------------------------------------------")

# 5. GRAFICAR (La clave para evitar el zigzag)
plt.figure(figsize=(10, 6))

# A) Dibujamos los puntos reales de prueba como una nube (en azul)
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Datos Reales (Prueba)')

# B) Creamos una secuencia de 100 puntos matemáticamente ORDENADOS para la línea
X_rango = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# C) Transformamos esos puntos ordenados a polinomio y predecimos sus valores
X_rango_poly = poly.transform(X_rango)
y_curva = modelo.predict(X_rango_poly)

# D) Dibujamos la curva usando exclusivamente los puntos ordenados (en rojo)
plt.plot(X_rango, y_curva, color='red', linewidth=3, label='Curva Polinómica')

# Detalles estéticos del gráfico
plt.title('Regresión Polinómica: Ingreso Medio vs Precio de Vivienda')
plt.xlabel('Ingreso Medio (Avg. Area Income)')
plt.ylabel('Precio (Price)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Mostrar el gráfico final
plt.show()