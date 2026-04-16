import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. CARGA DE DATOS (Ruta local para VS Code)
try:
    df = pd.read_csv('heart_disease_uci.csv')
    print("¡Archivo cargado con éxito desde la carpeta local!")
except FileNotFoundError:
    print("Error: No se encontró 'heart_disease_uci.csv'. Verifica que esté en la misma carpeta.")
    exit()

# Creamos la variable objetivo (0: Sano, 1: Enfermo)
df['target'] = (df['num'] > 0).astype(int)

# Definir predictores (X) y objetivo (y)
X = df.drop(columns=['id', 'dataset', 'num', 'target'], errors='ignore')
y = df['target']

# 2. CONFIGURAR PREPROCESAMIENTO
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. CREAR Y ENTRENAR EL MODELO (Pipeline)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000, solver='liblinear'))])

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# 4. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
y_pred = clf.predict(X_test)

print("\n--- Métricas de Evaluación ---")
print(f"Accuracy (Exactitud): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision (Precisión): {precision_score(y_test, y_pred):.2f}")
print(f"Recall (Sensibilidad): {recall_score(y_test, y_pred):.2f}")

# Generar la Matriz de Confusión (Gráfico)
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sano', 'Enfermo'], yticklabels=['Sano', 'Enfermo'])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión - Equipo Clasificación')
plt.show()

# 5. HERRAMIENTA DE PREDICCIÓN INTERACTIVA (Como en tu Foto 2)
print("\n--- Predicción Personalizada ---")
print("(Ingresa los datos o presiona Enter para usar valores por defecto)")

try:
    age = float(input("Edad: ") or 50)
    sex = input("Sexo (Male/Female): ") or "Male"
    cp = input("Dolor Torácico (typical angina, asymptomatic, non-anginal, atypical angina): ") or "asymptomatic"
    trestbps = float(input("Presión Arterial (ej. 120): ") or 120)
    chol = float(input("Colesterol (ej. 200): ") or 200)
    fbs = input("Azúcar > 120 (True/False): ") or "False"
    restecg = input("Electro (normal, lv hypertrophy, st-t wave abnormality): ") or "normal"
    thalch = float(input("Frecuencia Cardiaca Max (ej. 150): ") or 150)
    exang = input("Angina ejercicio (True/False): ") or "False"
    oldpeak = float(input("Depresión ST (ej. 1.0): ") or 1.0)
    slope = input("Pendiente ST (upsloping, flat, downsloping): ") or "flat"
    ca = float(input("Vasos coloreados (0-3): ") or 0)
    thal = input("Thal (normal, fixed defect, reversable defect): ") or "normal"

    # Crear DataFrame con la entrada del usuario
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]], 
                            columns=X.columns)
    
    prob = clf.predict_proba(input_df)[0]
    res = clf.predict(input_df)[0]

    print(f"\n>>> RESULTADO FINAL: {'PACIENTE ENFERMO' if res == 1 else 'PACIENTE SANO'}")
    print(f">>> Probabilidad de Enfermedad: {prob[1]:.2%}")

except ValueError:
    print("\nError: Ingresaste un texto en un campo numérico. Inténtalo de nuevo.")