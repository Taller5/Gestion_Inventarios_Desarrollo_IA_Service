# Analizador y limpiador de ventas_supermercado.csv
import pandas as pd

df = pd.read_csv("dataset/ventas_supermercado.csv", encoding="utf-8")

print("----- Análisis de calidad de datos -----")
print("\n1. Filas con valores nulos:")
print(df[df.isnull().any(axis=1)])

print("\n2. Filas duplicadas:")
print(df[df.duplicated()])

print("\n3. Tipos de datos por columna:")
print(df.dtypes)

print("\n4. Valores negativos en columnas numéricas:")
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    negativos = df[df[col] < 0]
    if not negativos.empty:
        print(f"\nColumna '{col}' tiene valores negativos:")
        print(negativos)

print("\n5. Resumen general:")
print(df.describe(include='all'))