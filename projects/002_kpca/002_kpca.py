import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    # Carga el conjunto de datos del archivo CSV
    df_heart = pd.read_csv('./data/raw/heart.csv')
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

    # Crea un objeto KernelPCA con 4 componentes principales y kernel polinómico
    kpca = KernelPCA(n_components=4, kernel='poly')
    # Ajusta el modelo KernelPCA a los datos de entrenamiento
    kpca.fit(X_train)

    # Transforma los datos de entrenamiento y prueba utilizando el modelo KernelPCA
    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)
    print("After KPCA")
    print(df_train.shape)
    print(df_test.shape)

    # Crea un objeto LogisticRegression y ajusta el modelo a los datos transformados
    logistic = LogisticRegression(solver='lbfgs')
    # Ajusta el modelo a los datos de entrenamiento
    logistic.fit(df_train, y_train)
    # Evalúa el modelo en los datos de prueba y muestra la puntuación
    print("Score KPCA", logistic.score(df_test, y_test))

    #Grafica el resultado de la regresion logistica
    plt.figure(figsize=(8,6))
    plt.scatter(df_train[:,0], df_train[:,1], c=y_train, cmap='viridis', edgecolor='k', s=100)
    plt.title('Kernel PCA - Logistic Regression', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.colorbar(label='Target')
    plt.grid()
    plt.show()





    



 