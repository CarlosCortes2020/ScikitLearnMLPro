import pandas as pd
from sklearn.linear_model import (
    RANSACRegressor,HuberRegressor
)
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Cargar el set de datos corrupto
    df = pd.read_csv('./data/raw/felicidad_corrupt.csv')
    # Muestra de las primeras filas del DataFrame
    print("Primeras filas del DataFrame:")
    print(df.head())

    # Separar características y variable objetivo
    X = df.drop(['country', 'score'], axis=1)
    y = df['score']

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Conjunto de entrenamiento: {X_train.shape}, Conjunto de prueba: {X_test.shape}")

    #Crearemos un diccionario para almacenar los estimadores y sus nombres
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for nombre, estimador in estimadores.items():
        print('='*64)
        # Entrenar el modelo
        estimador.fit(X_train, y_train)
        # Realizar predicciones
        y_pred = estimador.predict(X_test)
        # Evaluar el modelo utilizando el error cuadrático medio
        mse = mean_squared_error(y_test, y_pred)
        print(f"{nombre} Mean Squared Error: {mse}")


