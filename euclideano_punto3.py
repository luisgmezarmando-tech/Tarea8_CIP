import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------
def cargar_datos(ruta):
    """
    Lee el archivo CSV sin encabezados.
    Separa las primeras 784 columnas (X) de la etiqueta de clase final (y).
    """
    df = pd.read_csv(ruta, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y

# ---------------------------------------------------------
# 2. ENTRENAMIENTO (CÁLCULO DE CENTROIDES)
# ---------------------------------------------------------
def entrenar_euclidiano(X_train, y_train):
    """Calcula el centroide (vector media de 784 dimensiones) por clase."""
    centroides = {}
    clases_unicas = np.unique(y_train)
    
    for clase in clases_unicas:
        filas_clase = X_train[y_train == clase]
        centroides[clase] = filas_clase.mean(axis=0)
        
    return centroides

# ---------------------------------------------------------
# 3. EVALUACIÓN Y REPORTE
# ---------------------------------------------------------
def evaluar_modelo(X_test, y_test, centroides):
    filas_reporte = []
    
    # Clases ordenadas (2 y 3)
    clases = sorted(list(centroides.keys()))
    
    for i, x in enumerate(X_test):
        # A) Distancia euclidiana del patrón a cada centroide
        distancias = {}
        for clase in clases:
            distancias[clase] = np.linalg.norm(x - centroides[clase])
            
        # B) Predicción: clase con la distancia mínima
        prediccion = min(distancias, key=distancias.get)
        clase_real = y_test[i]
        
        # C) Acierto o Error
        resultado = "ACIERTO" if prediccion == clase_real else "ERROR"
            
        # D) Fila de reporte estandarizada
        fila = [
            i + 1,
            int(clase_real),
            np.round(distancias[2], 4),
            np.round(distancias[3], 4),
            int(prediccion),
            resultado
        ]
        filas_reporte.append(fila)
        
    # Construcción de la tabla
    columnas = [
        'Patrón', 
        'Clase Real', 
        'Distancia al centroide 2', 
        'Distancia al centroide 3', 
        'Clase asignada', 
        'Resultado'
    ]
    df_reporte = pd.DataFrame(filas_reporte, columns=columnas)
    df_reporte.to_csv("Tabla_Aciertos_Errores_MNIST.csv", index=False)
    
    # E) Accuracy
    total_aciertos = (df_reporte['Resultado'] == "ACIERTO").sum()
    accuracy = (total_aciertos / len(X_test)) * 100
    
    return df_reporte, accuracy

# ---------------------------------------------------------
# BLOQUE PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    ruta_train = '3_Luis_train.csv'
    ruta_test = '3_Luis_test.csv'
    
    print("Cargando conjuntos de datos...")
    X_train, y_train = cargar_datos(ruta_train)
    X_test, y_test = cargar_datos(ruta_test)
    
    print(f"Patrones de entrenamiento: {len(X_train)} (Clase 2: {np.sum(y_train==2)}, Clase 3: {np.sum(y_train==3)})")
    print(f"Patrones de prueba:        {len(X_test)} (Clase 2: {np.sum(y_test==2)}, Clase 3: {np.sum(y_test==3)})")
    
    # --- PASO 1: Entrenamiento ---
    centroides = entrenar_euclidiano(X_train, y_train)
    print("\nCentroides calculados correctamente para las clases 2 y 3.")
    
    # --- PASO 2: Evaluación ---
    df_res, accuracy = evaluar_modelo(X_test, y_test, centroides)
    
    print(f"\nAccuracy del clasificador: {accuracy:.2f}%")
    print("\nPrimeros 5 patrones evaluados en prueba:")
    print(df_res.head())
    
    # --- PASO 3: Matriz de Confusión (Clase 2 = Positive, Clase 3 = Negative) ---
    TP = int(((df_res['Clase Real'] == 2) & (df_res['Clase asignada'] == 2)).sum())
    FN = int(((df_res['Clase Real'] == 2) & (df_res['Clase asignada'] == 3)).sum())
    FP = int(((df_res['Clase Real'] == 3) & (df_res['Clase asignada'] == 2)).sum())
    TN = int(((df_res['Clase Real'] == 3) & (df_res['Clase asignada'] == 3)).sum())
    #Imprimir resultados de la matriz de confusión
    print("\n" + "-"*50)
    print("MATRIZ DE CONFUSIÓN (Convención: 2 = Positive, 3 = Negative)")
    print("-"*50)
    print(f"{'':22} {'Pred Positive (2)':<20} {'Pred Negative (3)':<20}")
    print(f"{'Real Positive (2)':<22} {'TP = ' + str(TP):<20} {'FN = ' + str(FN):<20}")
    print(f"{'Real Negative (3)':<22} {'FP = ' + str(FP):<20} {'TN = ' + str(TN):<20}")
    print("-"*50)
    print(f"Archivo de detalle generado: Tabla_Aciertos_Errores_MNIST.csv")