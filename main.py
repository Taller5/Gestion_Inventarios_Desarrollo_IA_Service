import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import os
import numpy as np
from fastapi.middleware.cors import CORSMiddleware # 1. Importar el middleware
from typing import List
from calendar import monthrange


# --- 1. Configuración Inicial y Carga del Modelo ---
app = FastAPI(title="API de Predicción de Ventas de Supermercado")

# 2. Definir los orígenes permitidos
# Permitimos localhost:5173 (tu frontend) y 127.0.0.1:8001 (FastAPI/Swagger)
PRODUCTION_FRONTEND_URL = "https://gestion-inventarios-desarrollo-fron.vercel.app/"

origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",  
    PRODUCTION_FRONTEND_URL,
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

# 3. Aplicar el Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Lista de orígenes permitidos
    allow_credentials=True,         # Permite cookies/headers de autenticación
    allow_methods=["*"],            # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],            # Permite todos los encabezados
)


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
    fecha_prediccion: datetime.date  # Formato YYYY-MM-DD
    precio_de_venta_esperado: float # Ahora es obligatorio
    promocion_activa: int # 0 o 1, obligatorio ya que el modelo fue entrenado con él

class PrediccionAnualInput(BaseModel):
    id_products: int
    precio_de_venta_esperado: float
    promocion_activa: int
    anios: List[int]  # Lista de años, máximo 3
    
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
            'precio_promedio_diario': [data.precio_de_venta_esperado],
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

@app.post("/predict_anual_quantity")
def predecir_cantidad_anual(data: PrediccionAnualInput):
    """
    Predice la cantidad total vendida por mes para los años seleccionados (máx 3).
    Devuelve una lista con {"mes": "YYYY-MM", "cantidad": total_predicho}
    """
    if MODELO_VENTAS is None:
        raise HTTPException(status_code=503, detail="El modelo de predicción no está disponible. Ejecute el worker de entrenamiento.")

    if not isinstance(data.anios, list) or len(data.anios) == 0 or len(data.anios) > 3:
        raise HTTPException(status_code=400, detail="El campo 'años' debe ser una lista de 1 a 3 años.")

    try:
        resultados = []

        for year in data.anios:
            for month in range(1, 13):
                dias_en_mes = monthrange(year, month)[1]
                fechas_mes = [datetime.date(year, month, dia) for dia in range(1, dias_en_mes + 1)]

                input_data = pd.DataFrame({
                    'id_products': [data.id_products] * dias_en_mes,
                    'dia_semana': [fecha.weekday() for fecha in fechas_mes],
                    'mes': [month] * dias_en_mes,
                    'trimestre': [(month - 1) // 3 + 1] * dias_en_mes,
                    'precio_promedio_diario': [data.precio_de_venta_esperado] * dias_en_mes,
                    'promocion_activa': [data.promocion_activa] * dias_en_mes,
                })

                predicciones = MODELO_VENTAS.predict(input_data)
                total_mes = int(max(0, round(np.sum(predicciones))))

                resultados.append({
                    "mes": f"{year}-{month:02d}",
                    "cantidad": total_mes
                })

        return resultados

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción anual. Detalle: {str(e)}")

# Para ejecutar el API: uvicorn main:app --reload --port 8001