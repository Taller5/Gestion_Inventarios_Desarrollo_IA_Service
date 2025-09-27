import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from sqlalchemy import create_engine # Necesitas instalar 'mysqlclient' o 'pymysql'
import numpy as np

# --- 1. SIMULACIÓN DE EXTRACCIÓN DE DATOS PLANOS (En producción, esto sería una consulta DB) ---
def obtener_datos_planos_de_db():
    # En un entorno real, usarías SQL Alchemy y la consulta JSON_TABLE.
    # engine = create_engine("mysql+pymysql://user:pass@host:port/dbname")
    # query = "SELECT i.date_invoices, ... [LA CONSULTA JSON_TABLE COMPLETA]"
    # df = pd.read_sql(query, engine)
    
    # Aquí usamos tus datos de prueba para simular el resultado de la consulta SQL
    # Asume que los datos están planos y listos
    df = pd.read_csv("dataset/ventas_supermercado.csv", encoding="utf-8") 
    df['date_invoices'] = pd.to_datetime(df['date_invoices']) # Convierte la columna date_invoices en una datetime
    return df

# --- 2. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
def entrenar_y_guardar_modelo():
    print("Iniciando Worker de Entrenamiento...")
    df = obtener_datos_planos_de_db()
    
    if df.empty:
        print("No hay datos para entrenar. Terminando.")
        return

    # A. Feature Engineering (Creación de características de tiempo)
    df['dia_semana'] = df['date_invoices'].dt.dayofweek
    df['mes'] = df['date_invoices'].dt.month
    df['trimestre'] = df['date_invoices'].dt.quarter
    
    # B. Preparación de X (Features) y Y (Target)
    FEATURES = ['id_products', 'dia_semana', 'mes', 'trimestre']
    TARGET = 'cantidad_vendida' # La variable objetivo que queremos predecir

    X = df[FEATURES] #datos de entrada
    Y = df[TARGET] #lo que se quiere predecir

    # C. División (Aunque en producción a menudo se entrena con todos los datos recientes)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # D. Entrenamiento del Modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, Y_train)
    
    # Opcional: Evaluar el modelo
    score = modelo.score(X_test, Y_test)
    print(f"Modelo entrenado con score R2 de: {score:.2f}")

    # E. Guardar el Modelo para FastAPI
    joblib.dump(modelo, "modelo/modelo_ventas.joblib")
    print("Modelo guardado correctamente en 'modelo/modelo_ventas.joblib'")

if __name__ == '__main__':
    import os
    os.makedirs('modelo', exist_ok=True)
    entrenar_y_guardar_modelo()

# Para ejecutar: python train_worker.py