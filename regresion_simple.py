import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset A [cite: 10]
df = pd.read_csv('USA_Housing.csv')

# 2. Justificación de la variable independiente [cite: 24]
# Filtramos solo columnas numéricas para la correlación
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlaciones = numeric_df.corr()['Price'].sort_values(ascending=False)
print("Correlación con el Precio:\n", correlaciones)

# Seleccionamos la de mayor correlación (ej. Avg. Area Income)
X = df[['Avg. Area Income']] 
y = df['Price']

# 3. Entrenar el modelo 
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Calcular métricas: MSE y R^2 
y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nError Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente R^2: {r2}")

# 5. Graficar la línea de tendencia 
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.3, label='Datos Reales')
plt.plot(X, y_pred, color='red', linewidth=2, label='Línea de Tendencia')
plt.title('Regresión Lineal Simple: Precio vs Ingreso Promedio')
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.legend()
plt.show()