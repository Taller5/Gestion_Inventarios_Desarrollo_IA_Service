import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine # Necesitas instalar 'mysqlclient' o 'pymysql'
import numpy as np
import os # Necesario para crear la carpeta 'modelo' y las rutas
import pymysql
from typing import Optional # Para mejor tipado

# ----------------------------------------------------
# 1. CONFIGURACIÓN DE LA BASE DE DATOS
# ----------------------------------------------------
#  ¡REEMPLAZA ESTOS VALORES CON TUS CREDENCIALES REALES! 
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "127.0.0.1"       # Ejemplo: "localhost" o la IP del servidor
DB_PORT = "3306"            # Puerto estándar de MySQL
DB_NAME = "gestion-inventarios-desarrollo-back"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- 1. Consulta con SQL para obtener los datos ---
def obtener_datos_planos_de_db() -> pd.DataFrame:
    """
    Se conecta a la base de datos, ejecuta la consulta SQL y retorna un DataFrame.
    """
    print("Conectando a la base de datos y ejecutando consulta...")
    
    try:
        # Crea el motor de conexión
        engine = create_engine(DATABASE_URL)
        
        # Define la consulta SQL para obtener los datos históricos necesarios.
        # ASEGÚRATE de que los nombres de las columnas coincidan exactamente.
        query = """
        SELECT 
            -- 1. Fecha de la factura
            i.date AS date_invoices,
            
            -- 2. El ID del producto (el campo 'id' de la tabla products)
            p.id AS id_products, 
            
            -- 3. Campos extraídos y transformados del JSON
            p_data.quantity AS cantidad_vendida,
            
            -- Lógica para convertir el descuento (discount > 0 significa promoción activa = 1)
            CASE 
                WHEN p_data.discount > 0 THEN 1 
                ELSE 0 
            END AS promocion_activa,
            
            -- Precio unitario de la venta
            p_data.price
            
        FROM 
            -- Tabla de facturas principal
            invoices i
            
            -- 1. Utiliza JSON_TABLE para aplanar el array JSON 'products'
            JOIN JSON_TABLE(
                i.products, 
                '$[*]' 
                COLUMNS (
                    -- Extrae el código (el campo de enlace)
                    product_code_json VARCHAR(50) PATH '$.code', 
                    -- Extrae la cantidad vendida
                    quantity INT PATH '$.quantity',
                    -- Extrae el valor del descuento
                    discount INT PATH '$.discount',
                    -- Extrae el precio
                    price DECIMAL(10, 2) PATH '$.price'
                )
            ) AS p_data
            
            -- 2. Une con la tabla 'products' usando el código extraído como enlace
            JOIN products p ON p_data.product_code_json = p.id
            
        WHERE 
            -- Filtrar por datos recientes (ej: último año)
            i.date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR);
        """
        
        # Carga los datos directamente en un DataFrame de Pandas
        df = pd.read_sql(query, engine)
        
        # Verifica si hay datos
        if df.empty:
            print("ADVERTENCIA: La consulta no devolvió datos.")
            return pd.DataFrame()

        # Conversión de tipos de datos (IMPORTANTE)
        df['date_invoices'] = pd.to_datetime(df['date_invoices']) 
        print(f"✅ Datos extraídos correctamente. Filas: {len(df)}")
        
        return df

    except Exception as e:
        print(f"❌ ERROR al conectar o consultar la DB: {e}")
        # En caso de error, puedes optar por cargar el CSV de simulación si existe
        # para probar la lógica sin la DB. (Descomenta las líneas de abajo para la simulación)
        # try:
        #     df_sim = pd.read_csv("dataset/ventas_supermercado.csv", encoding="utf-8")
        #     df_sim['date_invoices'] = pd.to_datetime(df_sim['date_invoices'])
        #     print("Usando datos de simulación por error de DB.")
        #     return df_sim
        # except FileNotFoundError:
        #     return pd.DataFrame()
        return pd.DataFrame() # Retorna vacío si falla



    # En un entorno real, usarías SQL Alchemy y la consulta JSON_TABLE.
    # engine = create_engine("mysql+pymysql://user:pass@host:port/dbname")
    # query = "SELECT i.date_invoices, ... [LA CONSULTA JSON_TABLE COMPLETA]"
    # df = pd.read_sql(query, engine)
    

# --- 2. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO (Para generar el modelo .joblib) ---
def entrenar_y_guardar_modelo():
    print("\nIniciando Worker de Entrenamiento...")
    df_plano_detallado = obtener_datos_planos_de_db()
    
    if df_plano_detallado.empty:
        print("No hay datos para entrenar. Terminando.")
        return

    # === PASO CRUCIAL: Agrupación y Suma (Creación de Features y Target Diario) ===
    
    # 1. Agrupar la demanda por Producto y por Día de Factura
    df_agregado = df_plano_detallado.groupby(
        ['id_products', 'date_invoices']
    ).agg(
        # Sumamos las cantidades vendidas (Target)
        cantidad_vendida_diaria=('cantidad_vendida', 'sum'),
        # FEATURE CLAVE: Calculamos el precio promedio del día 
        precio_promedio_diario=('precio', 'mean'), 
        # FEATURE OPCIONAL: Máximo de la promoción para el día (1 si hubo promoción)
        promocion_max=('promocion_activa', 'max'),
    ).reset_index()
    
    # Renombramos el Target y la feature promocion para asegurar el nombre exacto
    df_agregado.rename(columns={'cantidad_vendida_diaria': 'cantidad_vendida',
                                'promocion_max': 'promocion_activa'}, inplace=True) 

    # A. Feature Engineering (Creación de características de tiempo a partir de la fecha)
    df_agregado['dia_semana'] = df_agregado['date_invoices'].dt.dayofweek
    df_agregado['mes'] = df_agregado['date_invoices'].dt.month
    df_agregado['trimestre'] = df_agregado['date_invoices'].dt.quarter
    
    # B. Preparación de X (Features) y Y (Target)
    # ¡ESTAS FEATURES deben coincidir EXACTAMENTE con las usadas en la predicción (main.py)!
    FEATURES = ['id_products', 'dia_semana', 'mes', 'trimestre', 'precio_promedio_diario', 'promocion_activa']
    TARGET = 'cantidad_vendida' # La demanda total diaria
    
    X = df_agregado[FEATURES]
    Y = df_agregado[TARGET]

    # C. División (Usamos una parte para entrenamiento y otra para prueba)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # D. Entrenamiento del Modelo (usando el algoritmo Random Forest)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, Y_train)
    
    # Opcional: Evaluar el modelo
    score = modelo.score(X_test, Y_test)
    print(f"Modelo entrenado con score R2 de: {score:.2f}")

    # E. Guardar el Modelo
    os.makedirs('modelo', exist_ok=True) # Asegura que la carpeta exista
    joblib.dump(modelo, "modelo/modelo_ventas.joblib")
    print("Modelo guardado correctamente en 'modelo/modelo_ventas.joblib'")


# --- 4. EJECUCIÓN DEL SCRIPT ---
if __name__ == '__main__':
    # Asegura que la carpeta 'modelo' exista
    os.makedirs('modelo', exist_ok=True) 
    
    # Ejecuta la función de entrenamiento
    entrenar_y_guardar_modelo()

# Para ejecutar: python train_worker.py