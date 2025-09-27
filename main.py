import pandas as pd
from sklearn.model_selection import train_test_split
from fastapi import FastAPI

app = FastAPI()

@app.get("/train")
def train_model():
    # Cargar el CSV
    df = pd.read_csv("dataset/ventas_supermercado.csv")

    # Dividir en entrenamiento y prueba
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Guardar en archivos separados
    train.to_csv("ventas_train.csv", index=False)
    test.to_csv("ventas_test.csv", index=False)

    return {
        "Entrenamiento": len(train),
        "Prueba": len(test)
    }


