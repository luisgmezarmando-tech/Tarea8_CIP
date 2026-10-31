import numpy as np
import pandas as pd

# =====================================================================
# 1. CARGA DE DATOS
# =====================================================================
nombres_cols = [f'c{i}' for i in range(9)] + ['Class']
DOMINIO_VALORES = ['x', 'o', 'b']

df_E = pd.read_csv("4_Luis_train.csv", header=None, names=nombres_cols)
df_P = pd.read_csv("4_Luis_test.csv", header=None, names=nombres_cols)

print(f"Patrones en Entrenamiento (E): {len(df_E)}")
print(f"Patrones en Prueba (P):        {len(df_P)}")

# =====================================================================
# 2. ENTRENAMIENTO DEL ÁRBOL ID3
# =====================================================================
def calcular_entropia(columna_objetivo):
    """Calcula la entropía de Shannon de un conjunto."""
    total = len(columna_objetivo)
    if total == 0:
        return 0.0
    conteos = columna_objetivo.value_counts()
    entropia = 0.0
    for c in conteos:
        p = c / total
        entropia -= p * np.log2(p)
    return entropia

def calcular_ganancia(df, atributo, clase_objetivo='Class'):
    """Calcula la ganancia de información: H(S) - H(S|A)."""
    h_inicial = calcular_entropia(df[clase_objetivo])
    h_ponderada = 0.0
    
    for valor in DOMINIO_VALORES:
        sub = df[df[atributo] == valor]
        if len(sub) > 0:
            h_ponderada += (len(sub) / len(df)) * calcular_entropia(sub[clase_objetivo])
            
    return h_inicial - h_ponderada

def entrenar_id3(df, atributos_disponibles, clase_objetivo='Class', clase_padre='positive'):
    """Construye el árbol de decisión recursivamente."""
    if len(df) == 0:
        return clase_padre

    clases_unicas = df[clase_objetivo].unique()
    if len(clases_unicas) == 1:
        return clases_unicas[0]

    if len(atributos_disponibles) == 0:
        return df[clase_objetivo].mode()[0]

    ganancias = {attr: calcular_ganancia(df, attr, clase_objetivo) for attr in atributos_disponibles}
    mejor_atributo = max(ganancias, key=ganancias.get)

    if ganancias[mejor_atributo] == 0:
        return df[clase_objetivo].mode()[0]

    arbol = {mejor_atributo: {}}
    atributos_restantes = [a for a in atributos_disponibles if a != mejor_atributo]
    moda_actual = df[clase_objetivo].mode()[0]

    for valor in DOMINIO_VALORES:
        submatriz = df[df[mejor_atributo] == valor]
        arbol[mejor_atributo][valor] = entrenar_id3(
            submatriz, atributos_restantes, clase_objetivo, moda_actual
        )

    return arbol

atributos_iniciales = [f'c{i}' for i in range(9)]
arbol_entrenado = entrenar_id3(df_E, atributos_iniciales)
print("\nÁrbol ID3 entrenado con éxito.")

# =====================================================================
# 3. EXPORTACIÓN DEL ÁRBOL PARA VISUALIZACIÓN
# =====================================================================
def exportar_arbol_txt(arbol, ruta_archivo="ID3_entrenamiento.txt"):
    """Genera un archivo .txt con la estructura jerárquica ramificada."""
    lineas = []
    
    def recorrer(sub_arbol, prefijo=""):
        if not isinstance(sub_arbol, dict):
            return
        nodo = list(sub_arbol.keys())[0]
        ramas = list(sub_arbol[nodo].items())
        
        for idx, (val, hijo) in enumerate(ramas):
            es_ultimo = (idx == len(ramas) - 1)
            conector = "└── " if es_ultimo else "├── "
            
            if isinstance(hijo, dict):
                siguiente_nodo = list(hijo.keys())[0]
                lineas.append(f"{prefijo}{conector}[{nodo} == '{val}'] ──> (Evaluar {siguiente_nodo})")
                extension = "    " if es_ultimo else "│   "
                recorrer(hijo, prefijo + extension)
            else:
                lineas.append(f"{prefijo}{conector}[{nodo} == '{val}'] ──> [HOJA: {hijo}]")

    raiz = list(arbol.keys())[0]
    lineas.append(f"Raíz: {raiz}")
    recorrer(arbol)
    
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(f"Archivo de texto generado: {ruta_archivo}")

def exportar_arbol_dot(arbol, ruta_archivo="ID3_entrenamiento.dot"):
    """Genera un archivo .dot de Graphviz para visualizar el grafo en diagrama."""
    contador = [0]
    lineas = [
        "digraph ArbolID3 {",
        "    node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10];",
        "    edge [fontname=\"Helvetica\", fontsize=9];"
    ]
    
    def recorrer(sub_arbol):
        if not isinstance(sub_arbol, dict):
            return None
        nodo = list(sub_arbol.keys())[0]
        id_actual = f"nodo_{contador[0]}"
        contador[0] += 1
        
        lineas.append(f'    {id_actual} [label="{nodo}", fillcolor="#E3F2FD", color="#1565C0"];')
        
        for val, hijo in sub_arbol[nodo].items():
            if isinstance(hijo, dict):
                id_hijo = recorrer(hijo)
                lineas.append(f'    {id_actual} -> {id_hijo} [label="{val}"];')
            else:
                id_hoja = f"hoja_{contador[0]}"
                contador[0] += 1
                color = "#C8E6C9" if hijo == "positive" else "#FFCDD2"
                borde = "#2E7D32" if hijo == "positive" else "#C62828"
                lineas.append(f'    {id_hoja} [label="{hijo}", shape=ellipse, fillcolor="{color}", color="{borde}"];')
                lineas.append(f'    {id_actual} -> {id_hoja} [label="{val}"];')
                
        return id_actual

    recorrer(arbol)
    lineas.append("}")
    
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(f"Archivo Graphviz generado: {ruta_archivo}")

# Guardar los archivos de visualización
exportar_arbol_txt(arbol_entrenado, "ID3_entrenamiento.txt")
exportar_arbol_dot(arbol_entrenado, "ID3_entrenamiento.dot")

# =====================================================================
# 4. EVALUACIÓN SOBRE EL CONJUNTO DE PRUEBA
# =====================================================================
def clasificar_patron(arbol, patron, clase_fallback='positive'):
    """Navega el árbol recursivamente hasta una hoja."""
    if not isinstance(arbol, dict):
        return arbol

    nodo_actual = list(arbol.keys())[0]
    valor = patron[nodo_actual]

    if valor in arbol[nodo_actual]:
        siguiente = arbol[nodo_actual][valor]
        return clasificar_patron(siguiente, patron, clase_fallback)
    else:
        return clase_fallback

predicciones = []
resultados = []

for _, fila in df_P.iterrows():
    pred = clasificar_patron(arbol_entrenado, fila)
    real = fila['Class']
    predicciones.append(pred)
    resultados.append("ACIERTO" if pred == real else "ERROR")

df_P['Prediccion'] = predicciones
df_P['Resultado'] = resultados
df_P.to_csv("Tabla_Aciertos_Errores_TicTacToe.csv", index=False)

# =====================================================================
# 5. MATRIZ DE CONFUSIÓN Y EXACTITUD
# =====================================================================
TP = int(((df_P['Class'] == 'positive') & (df_P['Prediccion'] == 'positive')).sum())
FN = int(((df_P['Class'] == 'positive') & (df_P['Prediccion'] == 'negative')).sum())
FP = int(((df_P['Class'] == 'negative') & (df_P['Prediccion'] == 'positive')).sum())
TN = int(((df_P['Class'] == 'negative') & (df_P['Prediccion'] == 'negative')).sum())

accuracy = ((TP + TN) / len(df_P)) * 100

print("\n" + "-"*50)

print("MATRIZ DE CONFUSIÓN (Positive: x gana | Negative: x no gana)")
print("-"*50)
print(f"{'':22} {'Pred Positive':<18} {'Pred Negative':<18}")
print(f"{'Real Positive':<22} {'TP = ' + str(TP):<18} {'FN = ' + str(FN):<18}")
print(f"{'Real Negative':<22} {'FP = ' + str(FP):<18} {'TN = ' + str(TN):<18}")
print("-"*50)
print(f"Accuracy del modelo: {accuracy:.2f}%")