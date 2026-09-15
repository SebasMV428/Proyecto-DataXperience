import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LIMPIEZA DE DATOS
df = pd.read_csv('productos_tecno.csv')
df['Precio_Original'] = df['Precio_Original'].str.replace('$', '', regex=False).str.replace('.', '', regex=False)
df['Precio_Original'] = pd.to_numeric(df['Precio_Original'].str.strip(), errors='coerce')

print("--- ESTADÍSTICAS PARA TU DIAPOSITIVA 2 ---")
print(df['Precio_Original'].describe())

# 2. MODELO DE MACHINE LEARNING
X = df[['Precio_Original']] 
y = df['Unidades_Vendidas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
mae = mean_absolute_error(y_test, predicciones)

print("\n--- RESULTADOS ML PARA TU DIAPOSITIVA 4 ---")
print(f"Margen de error en unidades (MAE): {mae:.1f} unidades")
print(f"R2 Score: {r2_score(y_test, predicciones):.2f}")

# 3. GRÁFICOS PARA TU DIAPOSITIVA 3
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['Precio_Original'], color='purple')
plt.title('Distribución de Precios')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Precio_Original'], color='cyan')
plt.title('Valores Atípicos (Equipos Premium)')

plt.tight_layout()
plt.show() # Toma pantallazo a la ventana que se abre