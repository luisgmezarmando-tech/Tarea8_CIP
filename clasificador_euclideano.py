import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------
def cargar_datos(ruta):
    """Carga básica: Lee el CSV sin encabezados y separa X de y."""
    df = pd.read_csv(ruta, header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values

def limpiar_patrones(ruta):
    """Carga, verifica y purga patrones redundantes e indiscernibles."""
    df = pd.read_csv(ruta, header=None)
    features = df.columns[:-1].tolist()
    target = df.columns[-1]
    
    # Mostrar si hay duplicados
    duplicados = df[df.duplicated(keep=False)]
    if not duplicados.empty:
        print(f"[{ruta}] Patrones redundantes detectados.")
        
    # Contar cuántas clases distintas tiene cada grupo de características idénticas
    conteo_clases = df.groupby(features)[target].transform('nunique')
    
    # Filtro 1: Eliminar contradictorios (indiscernibles)
    df = df[conteo_clases == 1]
    
    # Filtro 2: Dejar solo 1 copia de los redundantes puros
    df = df.drop_duplicates(subset=features, keep='first')
        
    return df.iloc[:, :-1].values, df.iloc[:, -1].values

# ---------------------------------------------------------
# 2. ENTRENAMIENTO
# ---------------------------------------------------------
def entrenar_euclidiano(X_train, y_train):
    """Calcula el centroide (vector media) para cada clase presente."""
    centroides = {}
    clases_unicas = np.unique(y_train)
    
    for clase in clases_unicas:
        filas_clase = X_train[y_train == clase]
        centroides[clase] = filas_clase.mean(axis=0)
        
    return centroides

# ---------------------------------------------------------
# 3, 4 y 5. EVALUACIÓN Y REPORTE DINÁMICO
# ---------------------------------------------------------
def evaluar_modelo(X_test, y_test, centroides, incluir_atributos=False):
    """
    Evalúa el clasificador euclidiano.
    - Si incluir_atributos=False: Genera la tabla estándar de evaluación 
      (Patrón, Clase Real, Distancias a centroides, Clase asignada, Resultado).
    - Si incluir_atributos=True: Agrega además todas las características x_j.
    """
    filas_reporte = []
    clases = sorted(list(centroides.keys()))
    
    for i, x in enumerate(X_test):
        # A) Calcular distancias euclidianas a cada centroide 
        distancias = {c: np.linalg.norm(x - centroides[c]) for c in clases}
            
        # B) Clase asignada (mínima distancia)
        prediccion = min(distancias, key=distancias.get)
        clase_real = y_test[i]
        
        # C) Acierto o Error
        resultado = "ACIERTO" if prediccion == clase_real else "ERROR"
            
        # D) Construir la fila dinámicamente como diccionario
        fila = {
            'Patrón': i + 1,
            'Clase Real': int(clase_real) if isinstance(clase_real, (np.integer, int, float)) and clase_real.is_integer() else clase_real
        }
        
        # Si se desea incluir cada una de las características (x0, x1, ...)
        if incluir_atributos:
            for j, val in enumerate(x):
                fila[f'x{j}'] = np.round(val, 4)
        
        # Agregar columnas de distancia según las clases existentes
        for c in clases:
            nombre_clase = int(c) if isinstance(c, (np.integer, int, float)) and float(c).is_integer() else c
            fila[f'Distancia al centroide {nombre_clase}'] = np.round(distancias[c], 4)
            
        fila['Clase asignada'] = int(prediccion) if isinstance(prediccion, (np.integer, int, float)) and float(prediccion).is_integer() else prediccion
        fila['Resultado'] = resultado
        
        filas_reporte.append(fila)
        
    # Pandas genera la tabla y columnas en el orden exacto de inserción
    df_reporte = pd.DataFrame(filas_reporte)
    df_reporte.to_csv("Tabla_Aciertos_Errores.csv", index=False)
    
    # E) Accuracy
    total_aciertos = (df_reporte['Resultado'] == "ACIERTO").sum()
    accuracy = (total_aciertos / len(X_test)) * 100
    
    return df_reporte, accuracy

# ---------------------------------------------------------
# BLOQUE PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    ruta_train = 'spambase_E.csv'
    ruta_test = 'spambase_P.csv'
    
    usar_limpieza = False 
    
    # --- PASO 1: Carga ---
    if usar_limpieza:
        X_train, y_train = limpiar_patrones(ruta_train)
        X_test, y_test = limpiar_patrones(ruta_test)
    else:
        X_train, y_train = cargar_datos(ruta_train)
        X_test, y_test = cargar_datos(ruta_test)
    
    # --- PASO 2: Entrenamiento ---
    centroides = entrenar_euclidiano(X_train, y_train)
    
    print("\n=== CENTROIDES CALCULADOS ===")
    for clase, mu in centroides.items():
        print(f"Clase {clase}: {np.round(mu, 4)}")
        
    # --- Evaluación y Reporte ---

    df_res, accuracy = evaluar_modelo(X_test, y_test, centroides, incluir_atributos=False)
    
    print(f"\nAccuracy del clasificador: {accuracy:.2f}%")
    print(f"Patrones evaluados: {len(df_res)}")
    
    # Muestra de las primeras y últimas filas
    print("\nPrimeros 5 patrones evaluados:")
    print(df_res.head())
    
    # Matriz de Confusión (Clase 1 = Positive, Clase 0 = Negative)
    TP = ((df_res['Clase Real'] == 1) & (df_res['Clase asignada'] == 1)).sum()
    FN = ((df_res['Clase Real'] == 1) & (df_res['Clase asignada'] == 0)).sum()
    FP = ((df_res['Clase Real'] == 0) & (df_res['Clase asignada'] == 1)).sum()
    TN = ((df_res['Clase Real'] == 0) & (df_res['Clase asignada'] == 0)).sum()
    
    print("\n=== MATRIZ DE CONFUSIÓN ===")
    print(f"TP: {TP} | FN: {FN}")
    print(f"FP: {FP} | TN: {TN}")