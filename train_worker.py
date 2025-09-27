import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from sqlalchemy import create_engine # Necesitas instalar 'mysqlclient' o 'pymysql'
import numpy as np
import os # Necesario para crear la carpeta 'modelo' y las rutas

# --- 1. SIMULACIÓN DE EXTRACCIÓN DE DATOS PLANOS (En producción, esto sería una consulta DB) ---
def obtener_datos_planos_de_db():
    # En un entorno real, usarías SQL Alchemy y la consulta JSON_TABLE.
    # engine = create_engine("mysql+pymysql://user:pass@host:port/dbname")
    # query = "SELECT i.date_invoices, ... [LA CONSULTA JSON_TABLE COMPLETA]"
    # df = pd.read_sql(query, engine)
    
    # Aquí usamos tus datos de prueba para simular el resultado de la consulta SQL
    # Asegúrate de que este archivo exista en tu carpeta 'dataset/'
    try:
        df = pd.read_csv("dataset/ventas_supermercado.csv", encoding="utf-8") 
    except FileNotFoundError:
        print("ERROR: No se encontró 'dataset/ventas_supermercado.csv'. Asegúrate de haber copiado y guardado los datos CSV.")
        return pd.DataFrame()
        
    df['date_invoices'] = pd.to_datetime(df['date_invoices']) # Convierte la columna date_invoices en una datetime
    return df

# --- 2. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO (Para generar el modelo .joblib) ---
def entrenar_y_guardar_modelo():
    print("Iniciando Worker de Entrenamiento...")
    df_plano_detallado = obtener_datos_planos_de_db()
    
    if df_plano_detallado.empty:
        print("No hay datos para entrenar. Terminando.")
        return

    # === PASO CRUCIAL: Agrupación y Suma (Creación del Target Diario) ===
    # El modelo de series de tiempo necesita la demanda total por día.
    
    # 1. Agrupar la demanda por Producto y por Día de Factura
    df_agregado = df_plano_detallado.groupby(
        ['id_products', 'date_invoices']
    ).agg(
        # Sumamos las cantidades vendidas para obtener el total diario por producto (Target)
        cantidad_vendida_diaria=('cantidad_vendida', 'sum'),
        # FEATURE CLAVE: Calculamos el precio promedio del día para alimentar el modelo
        precio_promedio_diario=('precio', 'mean'), 
        # FEATURE OPCIONAL: Mantenemos promocion_activa (el valor máximo, será 1 si hubo al menos una promoción)
        promocion_activa=('promocion_activa', 'max'),
    ).reset_index()
    
    # Renombramos el Target para mayor claridad
    df_agregado.rename(columns={'cantidad_vendida_diaria': 'cantidad_vendida',
                            'promocion_max': 'promocion_activa'}, inplace=True) 

    # A. Feature Engineering (Creación de características de tiempo a partir de la fecha)
    df_agregado['dia_semana'] = df_agregado['date_invoices'].dt.dayofweek
    df_agregado['mes'] = df_agregado['date_invoices'].dt.month
    df_agregado['trimestre'] = df_agregado['date_invoices'].dt.quarter
    
    # B. Preparación de X (Features) y Y (Target)
    # ¡ESTAS FEATURES deben coincidir EXACTAMENTE con las usadas en la predicción!
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


# --- 3. FUNCIÓN DE PREDICCIÓN (Esta la usarás en tu API para hacer la predicción) ---
def predecir_demanda(datos_nuevos_brutos: list) -> np.ndarray:
    """
    Realiza una predicción de demanda con el modelo guardado.
    Recibe una lista de diccionarios (datos brutos de los días futuros) y 
    devuelve la predicción.
    """
    print("Cargando modelo y preparando datos para predicción...")
    # 1. Cargar el modelo
    modelo = joblib.load("modelo/modelo_ventas.joblib")

    # 2. Preparar los datos de entrada para la predicción
    df_prediccion = pd.DataFrame(datos_nuevos_brutos) 
    
    # A. Feature Engineering (Debe ser IDÉNTICO al entrenamiento)
    df_prediccion['date_invoices'] = pd.to_datetime(df_prediccion['date_invoices'])

    # Creación de las features de tiempo
    df_prediccion['dia_semana'] = df_prediccion['date_invoices'].dt.dayofweek
    df_prediccion['mes'] = df_prediccion['date_invoices'].dt.month
    df_prediccion['trimestre'] = df_prediccion['date_invoices'].dt.quarter
    
    # Renombrar/Crear la columna de precio para que coincida con el entrenamiento
    # Asumimos que los datos de entrada traen la columna 'precio' que es el precio pronosticado/futuro
    if 'precio' in df_prediccion.columns:
        df_prediccion.rename(columns={'precio': 'precio_promedio_diario'}, inplace=True)
    else:
        # Esto generaría un error si falta la columna. Aquí se puede asignar un valor por defecto o lanzar una excepción.
        print("ADVERTENCIA: La columna 'precio' o 'precio_promedio_diario' no se encontró en los datos de predicción.")


    # B. Seleccionar las FEATURES exactas (deben tener el mismo nombre que en el entrenamiento)
    FEATURES = ['id_products', 'dia_semana', 'mes', 'trimestre', 'precio_promedio_diario', 'promocion_activa']
    X_pred = df_prediccion[FEATURES]
    
    # 3. Realizar la predicción
    predicciones = modelo.predict(X_pred)
    
    return predicciones


if __name__ == '__main__':
    # Asegura que la carpeta 'modelo' exista antes de guardar el archivo
    os.makedirs('modelo', exist_ok=True) 
    
    # Ejecuta la función de entrenamiento cuando corres el script directamente
    entrenar_y_guardar_modelo()

    # --- Ejemplo de Uso de la Función de Predicción (Opcional) ---
    # Nota: predecir_demanda debe recibir datos en formato de lista de dicts
    # datos_futuros = [
    #     {'id_products': 3, 'date_invoices': '2024-09-01', 'precio': 4000, 'promocion_activa': 0},
    #     {'id_products': 3, 'date_invoices': '2024-09-02', 'precio': 4000, 'promocion_activa': 1}
    # ]
    # try:
    #     pred = predecir_demanda(datos_futuros)
    #     print(f"\nPredicciones para los próximos días: {pred}")
    # except FileNotFoundError:
    #     print("No se pudo cargar el modelo para el ejemplo de predicción.")

# Para ejecutar el entrenamiento: python train_worker.py