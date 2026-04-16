import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Cargar el Dataset A [cite: 10]
df = pd.read_csv('USA_Housing.csv')

# 2. Selección de variables para 3D (3 variables independientes) 
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms']]
y = df['Price']

# 3. Machine Learning: División de datos [cite: 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenamiento del modelo [cite: 29]
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predicción y Métricas [cite: 29]
predictions = model.predict(X_test)
r2 = metrics.r2_score(y_test, predictions)
print(f"R^2 Score: {r2:.4f}")

# 6. GRÁFICO EN TRES DIMENSIONES (3D)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extraer columnas para el gráfico
x_data = X_test['Avg. Area Income']
y_data = X_test['Avg. Area House Age']
z_data = y_test  # Precio Real

# Graficar los puntos reales
scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis', marker='o', alpha=0.5, label='Datos Reales')

# Crear una malla para el plano de predicción
x_surf, y_surf = np.meshgrid(np.linspace(x_data.min(), x_data.max(), 20),
                             np.linspace(y_data.min(), y_data.max(), 20))

# Para el plano, fijamos la tercera variable (habitaciones) en su promedio
z_surf = (model.coef_[0] * x_surf + 
          model.coef_[1] * y_surf + 
          model.coef_[2] * X_test['Avg. Area Number of Rooms'].mean() + 
          model.intercept_)

# Graficar el plano de la regresión
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3, label='Plano de Predicción')

# Etiquetas de los ejes
ax.set_xlabel('Ingreso Promedio')
ax.set_ylabel('Edad de la Casa')
ax.set_zlabel('Precio de Vivienda')
plt.title('Regresión Múltiple en 3D: Predicción de Precios')

plt.show()