import pandas as pd
import sklearn

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Carga el dataset de indice de la felicidad
    df_happy = pd.read_csv('./data/raw/felicidad.csv')
    # Imprime las 10 primeras filas del dataset
    print(df_happy.head(10))
    # Imprime las caracteristicas del dataset
    print(df_happy.info())
    #Impprime las estadisticas descriptivas del dataset
    print(df_happy.describe())

    # Dividir el dataset en conjunto de entrenamiento y prueba
    X = df_happy[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = df_happy[['score']]
    print(X.shape)
    print(y.shape)

    # Dividir el dataset en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print('---------')
    # Modelo de regresion lineal
    model_linear = LinearRegression().fit(X_train, y_train)
    # Predicciones y evaluacion del modelo lineal
    y_predict_linear = model_linear.predict(X_test)
    # Evaluacion del modelo lineal, los valores mas bajos son mejores
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    # Imprime la perdida del modelo lineal
    print("Linear Loss: ", linear_loss)

    print('---------')
    # Modelo de regresion con regularizacion Lasso
    model_Lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    # Predicciones y evaluacion del modelo Lasso
    y_predict_Lasso = model_Lasso.predict(X_test)
    # Evaluacion del modelo Lasso, los valores mas bajos son mejores
    Lasso_loss = mean_squared_error(y_test, y_predict_Lasso)
    # Imprime la perdida del modelo Lasso
    print("Lasso Loss: ", Lasso_loss)

    print('---------')
    # Modelo de regresion con regularizacion Ridge
    model_Ridge = Ridge(alpha=1).fit(X_train, y_train)
    # Predicciones y evaluacion del modelo Ridge
    y_predict_Ridge = model_Ridge.predict(X_test)
    # Evaluacion del modelo Ridge, los valores mas bajos son mejores
    Ridge_loss = mean_squared_error(y_test, y_predict_Ridge)
    # Imprime la perdida del modelo Ridge
    print("Ridge Loss: ", Ridge_loss)

    print('---------')
    # Modelo de regresion con regularizacion ElasticNet
    model_ElasticNet = ElasticNet(alpha=0.02, l1_ratio=1).fit(X_train, y_train)
    # Predicciones y evaluacion del modelo ElasticNet
    y_predict_ElasticNet = model_ElasticNet.predict(X_test)
    # Evaluacion del modelo ElasticNet, los valores mas bajos son mejores
    ElasticNet_loss = mean_squared_error(y_test, y_predict_ElasticNet)
    # Imprime la perdida del modelo ElasticNet
    print("ElasticNet Loss: ", ElasticNet_loss)
    
    #Imprimir los coeficientes de cada modelo
    print('---------')
    print("Linear Coefficients: ", model_linear.coef_)
    print("Lasso Coefficients: ", model_Lasso.coef_)
    print("Ridge Coefficients: ", model_Ridge.coef_)
    print("ElasticNet Coefficients: ", model_ElasticNet.coef_)

    # Imprimir la interseccion de cada modelo
    print('---------')
    print("Linear Intercept: ", model_linear.intercept_)
    print("Lasso Intercept: ", model_Lasso.intercept_)
    print("Ridge Intercept: ", model_Ridge.intercept_)
    print("ElasticNet Intercept: ", model_ElasticNet.intercept_)

    


