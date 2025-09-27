import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import os
import numpy as np

# --- 1. Configuración Inicial y Carga del Modelo ---
app = FastAPI(title="API de Predicción de Ventas de Supermercado")

MODEL_PATH = "modelo/modelo_ventas.joblib"
MODELO_VENTAS = None

if os.path.exists(MODEL_PATH):
    MODELO_VENTAS = joblib.load(MODEL_PATH)
    print("✅ Modelo de ventas cargado y listo.")
else:
    print(f"❌ ADVERTENCIA: Modelo no encontrado en {MODEL_PATH}. Ejecute 'train_worker.py'.")

# --- 2. Definición del Esquema de Entrada (Input) ---
# Hemos añadido el campo 'promocion_activa' y ajustado el nombre de la fecha
class PrediccionInput(BaseModel):
    id_products: int
    fecha_prediccion: str  # Formato YYYY-MM-DD
    precio_de_venta_esperado: float # Ahora es obligatorio
    promocion_activa: int # 0 o 1, obligatorio ya que el modelo fue entrenado con él
    
# --- 3. Endpoint de Predicción ---
@app.post("/predict_quantity")
def predecir_cantidad_vendida(data: PrediccionInput):
    """
    Realiza una predicción de la cantidad a vender para un producto
    en una fecha futura específica.
    """
    if MODELO_VENTAS is None:
        raise HTTPException(status_code=503, detail="El modelo de predicción no está disponible. Ejecute el worker de entrenamiento.")

    try:
        # A. Feature Engineering (Debe replicar EXACTAMENTE lo que se hizo en train_worker.py)
        fecha = pd.to_datetime(data.fecha_prediccion)
        
        # El DataFrame de entrada debe tener EXACTAMENTE las mismas 6 columnas
        # y los mismos nombres que se usaron en el entrenamiento.
        input_data = pd.DataFrame({
            'id_products': [data.id_products],
            'dia_semana': [fecha.dayofweek],
            'mes': [fecha.month],
            'trimestre': [fecha.quarter],
            # ¡SOLUCIÓN!: Usamos el dato de entrada, pero lo nombramos EXACTAMENTE
            # como lo espera el modelo entrenado: 'precio_promedio_diario'
            'precio_promedio_diario': [data.precio_de_venta_esperado],
            # Necesitas la columna 'promocion_activa' si la usaste en el entrenamiento
            'promocion_activa': [data.promocion_activa], 
        })
        
        # B. Realizar la predicción
        prediccion_raw = MODELO_VENTAS.predict(input_data)[0]
        
        # C. Post-procesamiento: asegurar que la cantidad sea un entero no negativo
        cantidad_predicha = max(0, int(round(prediccion_raw)))

        return {
            "id_products": data.id_products,
            "fecha_prediccion": data.fecha_prediccion,
            "precio_usado": data.precio_de_venta_esperado,
            "cantidad_vendida_estimada": cantidad_predicha,
            "unidad": "unidades"
        }

    except Exception as e:
        # Esto te mostrará el error exacto de Pandas/Scikit-learn si hay otro problema
        raise HTTPException(status_code=500, detail=f"Error durante la predicción. Detalle: {str(e)}")

# Para ejecutar el API: uvicorn main:app --reload