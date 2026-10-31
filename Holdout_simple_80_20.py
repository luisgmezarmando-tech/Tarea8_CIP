import pandas as pd
import numpy as np

# 1. Leer el archivo spambase.data (local o directo desde la URL de UCI)
# Si lo tienes en la misma carpeta: 'spambase.data'
df = pd.read_csv("spambase.data", header=None)


# La última columna (índice 57) contiene las etiquetas (0 o 1)
col_clase = df.shape[1] - 1

# 2. Separar por clase manteniendo el orden de aparición original
clase_0 = df[df[col_clase] == 0]
clase_1 = df[df[col_clase] == 1]

# 3. Calcular el 80% exacto para cada clase
n_train_0 = int(len(clase_0) * 0.8)  # 2788 * 0.8 = 2230
n_train_1 = int(len(clase_1) * 0.8)  # 1813 * 0.8 = 1450

# 4. Partición abreviada (primeros 80% para E, últimos 20% para P)
E_0, P_0 = clase_0.iloc[:n_train_0], clase_0.iloc[n_train_0:]
E_1, P_1 = clase_1.iloc[:n_train_1], clase_1.iloc[n_train_1:]

# 5. Unir los subconjuntos
conjunto_E = pd.concat([E_0, E_1])
conjunto_P = pd.concat([P_0, P_1])

print(f"Conjunto E (Entrenamiento): {conjunto_E.shape[0]} patrones (Clase 0: {len(E_0)}, Clase 1: {len(E_1)})")
print(f"Conjunto P (Prueba):        {conjunto_P.shape[0]} patrones (Clase 0: {len(P_0)}, Clase 1: {len(P_1)})")

# 6. Guardar en archivos CSV listos para clasificar
conjunto_E.to_csv("spambase_E.csv", index=False, header=False)
conjunto_P.to_csv("spambase_P.csv", index=False, header=False)