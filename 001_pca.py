import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    # Carga el conjunto de datos del archivo CSV
    df_heart = pd.read_csv('./csv/heart.csv')
    # Muestra las primeras 5 filas del DataFrame
    print(df_heart.head(5)) 

    # Separa las características y el objetivo
    df_features = df_heart.drop(columns=['target'], axis=1)
    # Almacena el objetivo
    df_target = df_heart['target']

    # Estandariza las características
    df_features = StandardScaler().fit_transform(df_features)
    
    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    # Muestra las dimensiones de los conjuntos de datos
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #n_componentes = min(n_muestras, n_features)
    # Crea un objeto PCA
    pca = PCA(n_components=3)
    pca.fit(X_train)

    # Ajusta el modelo IncrementalPCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    # Muestra la varianza explicada por cada componente
    print("Explained variance PCA:", pca.explained_variance_)
    print("Explained variance IPCA:", ipca.explained_variance_)

    # Grafica la varianza explicada por cada componente
    plt.figure(figsize=(8,5))
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_, marker='o', label='PCA')
    plt.plot(range(len(ipca.explained_variance_)), ipca.explained_variance_, marker='o', label='IPCA')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.legend()
    plt.grid()
    plt.show()

    # Crea un modelo de regresión logística
    logistic = LogisticRegression(solver='lbfgs')
    
    df_train_pca = pca.transform(X_train)
    df_test_pca = pca.transform(X_test)
    logistic.fit(df_train_pca, y_train)
    print("Score PCA:", logistic.score(df_test_pca, y_test))

    df_train_ipca = ipca.transform(X_train)
    df_test_ipca = ipca.transform(X_test)
    logistic.fit(df_train_ipca, y_train)
    print("Score IPCA:", logistic.score(df_test_ipca, y_test))



    



 