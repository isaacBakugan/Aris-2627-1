# Nombre del integrante: Diego Goncalves
# Cédula del integrante: 30124687

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def sigmoide(z):
    # Clip para evitar el overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def derivada_sigmoide(a):
    return a * (1.0 - a)

class PerceptronMulticapa:
    def __init__(self, u, v, L, b):
        self.u = int(u)  # Entradas
        self.v = int(v)  # Salidas
        self.L = int(L)  # Capas ocultas
        self.b = int(b)  # Neuronas por capa oculta
        self.e = 0       # Épocas acumuladas
        self.pesos = []  # Lista de matrices de pesos
        self.inicializar_pesos()

    def inicializar_pesos(self):
        self.pesos = []
        # Capa que conecta 'u' con 'b'
        self.pesos.append(np.random.uniform(-0.5, 0.5, (self.b, self.u)))
        
        # Capas que conecta 'b' con 'b'
        for _ in range(self.L - 1):
            self.pesos.append(np.random.uniform(-0.5, 0.5, (self.b, self.b)))
            
        # Capa que conecta 'b' con 'v'
        self.pesos.append(np.random.uniform(-0.5, 0.5, (self.v, self.b)))

    def forward(self, X):
        # Propagación hacia adelante
        activaciones = [X]
        for W in self.pesos:
            Z = np.dot(activaciones[-1], W.T)
            A = sigmoide(Z)
            activaciones.append(A)
        return activaciones

    def predecir(self, X):
        activaciones = self.forward(X)
        salida = activaciones[-1]
        if self.v == 1:
            return (salida >= 0.5).astype(int).flatten()
        else:
            return np.argmax(salida, axis=1)

    def entrenar(self, X, y_onehot, epocas, lr=0.5):
        historial_error = []
        N = X.shape[0]

        for _ in range(epocas):
            # Propagación hacia adelante
            activaciones = self.forward(X)
            salida = activaciones[-1]

            # Error cuadrático medio (MSE)
            error = np.mean(0.5 * (salida - y_onehot) ** 2)
            historial_error.append(error)

            # Retropropagación
            deltas = [None] * len(self.pesos)
            
            # Delta de la capa de salida
            delta_salida = (salida - y_onehot) * derivada_sigmoide(salida)
            deltas[-1] = delta_salida

            # Deltas de capas ocultas hacia atrás
            for l in range(len(self.pesos) - 2, -1, -1):
                delta_l = np.dot(deltas[l + 1], self.pesos[l + 1]) * derivada_sigmoide(activaciones[l + 1])
                deltas[l] = delta_l

            # Actualización de pesos con gradiente descendente
            for l in range(len(self.pesos)):
                grad = np.dot(deltas[l].T, activaciones[l]) / N
                self.pesos[l] -= lr * grad

        self.e += epocas
        return historial_error

def guardar_red(ruta, red):
    c = max(red.u, red.b)
    with open(ruta, 'w', newline='') as f:
        writer = csv.writer(f)
        # Encabezado
        encabezado = ['tipo', 'capa', 'pos'] + [f'w{i}' for i in range(c)]
        writer.writerow(encabezado)

        # Hiperparámetros en las primeras 5 filas
        params = [('u', red.u), ('v', red.v), ('L', red.L), ('b', red.b), ('e', red.e)]
        for letra, val in params:
            fila = [letra, val, ''] + [''] * c
            writer.writerow(fila)

        # Capas ocultas
        for num_capa, W in enumerate(red.pesos[:-1], start=1):
            for pos_neurona, fila_pesos in enumerate(W):
                fila = ['h', num_capa, pos_neurona] + list(fila_pesos)
                # Rellenar con vacíos si tiene menos de c pesos
                fila += [''] * (c - len(fila_pesos))
                writer.writerow(fila)

        # Capa de salida
        capa_salida_idx = red.L + 1
        W_salida = red.pesos[-1]
        for pos_neurona, fila_pesos in enumerate(W_salida):
            fila = ['o', capa_salida_idx, pos_neurona] + list(fila_pesos)
            fila += [''] * (c - len(fila_pesos))
            writer.writerow(fila)

    print(f"\nRed guardada exitosamente en: {ruta}")

def cargar_red(ruta):
    with open(ruta, 'r') as f:
        reader = list(csv.reader(f))

    params = {}
    for i in range(1, 6):
        letra = reader[i][0].strip()
        val = int(reader[i][1].strip())
        params[letra] = val

    red = PerceptronMulticapa(params['u'], params['v'], params['L'], params['b'])
    red.e = params['e']

    for fila in reader[6:]:
        if not fila:
            continue
        tipo = fila[0].strip()
        capa = int(fila[1].strip())
        pos = int(fila[2].strip())
        
        if tipo == 'h':
            tam = red.u if capa == 1 else red.b
            idx_capa = capa - 1
        else:
            tam = red.b
            idx_capa = len(red.pesos) - 1

        pesos_neurona = [float(p) for p in fila[3:3 + tam]]
        red.pesos[idx_capa][pos] = pesos_neurona

    print(f"\nRed cargada exitosamente desde: {ruta}")
    return red

def resolver_ruta(archivo):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    en_assets = os.path.join(directorio_actual, "assets", archivo)
    en_script = os.path.join(directorio_actual, archivo)
    if os.path.exists(en_assets):
        return en_assets
    if os.path.exists(en_script):
        return en_script
    return archivo

def cargar_csv_datos(ruta, u_esperado, v_esperado):
    ruta_real = resolver_ruta(ruta)
    entradas = []
    esperados = []
    with open(ruta_real, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.split(',')
            try:
                vals = [float(x) for x in partes]
            except ValueError:
                continue  # Salta encabezados de texto si existen
            entradas.append(vals[:-1])
            esperados.append(int(vals[-1]))

    X = np.array(entradas)
    y = np.array(esperados, dtype=int)

    if np.min(y) == 1 and np.max(y) == v_esperado:
        y = y - 1
    elif np.min(y) == -1 and np.max(y) == 1 and v_esperado == 2:
        y = np.where(y == -1, 0, 1)

    if X.shape[1] != u_esperado:
        raise ValueError(f"Error de dimensión: El archivo tiene {X.shape[1]} entradas, "
                         f"pero la red está configurada para {u_esperado}. Configure la red con u = {X.shape[1]}.")

    clases_detectadas = len(np.unique(y))
    if v_esperado > 1 and clases_detectadas > v_esperado:
        raise ValueError(f"Error de clases: El archivo tiene {clases_detectadas} clases distintas, "
                         f"pero la red tiene v = {v_esperado}. Configure la red con v = {clases_detectadas}.")

    # Codificación One-Hot para el entrenamiento
    if v_esperado == 1:
        y_onehot = y.reshape(-1, 1).astype(float)
    else:
        y_onehot = np.zeros((len(y), v_esperado))
        for i, val in enumerate(y):
            y_onehot[i, val] = 1.0

    return X, y, y_onehot

def graficar_resultados(X, y_real, y_pred, modo, info_error=None, v=2):
    es_3d = X.shape[1] >= 3
    colores_coincidencia = ['green' if r == p else 'red' for r, p in zip(y_real, y_pred)]

    fig = plt.figure(figsize=(16, 4.5))

    # Gráfico de los datos esperados
    if es_3d:
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_real, cmap='tab10', s=40, edgecolors='k')
        ax1.set_xlabel("X1"); ax1.set_ylabel("X2"); ax1.set_zlabel("X3")
    else:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(X[:, 0], X[:, 1], c=y_real, cmap='tab10', s=50, edgecolors='k')
        ax1.set_xlabel("X1"); ax1.set_ylabel("X2")
    ax1.set_title("Datos Reales")

    # Gráfico de la coincidencia de la predicción
    if es_3d:
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=colores_coincidencia, s=40, edgecolors='k')
        ax2.set_xlabel("X1"); ax2.set_ylabel("X2"); ax2.set_zlabel("X3")
    else:
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(X[:, 0], X[:, 1], c=colores_coincidencia, s=50, edgecolors='k')
        ax2.set_xlabel("X1"); ax2.set_ylabel("X2")
    ax2.set_title("Predicción (Verde: Acierto, Rojo: Fallo)")

    # Gráfico del error de la red
    if modo == 'entrenamiento':
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(range(1, len(info_error) + 1), info_error, color='blue', lw=2)
        ax3.set_title("Error de la Red por Época")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("Error Cuadrático Medio (MSE)")
        ax3.grid(True)
    else:
        ax3 = fig.add_subplot(1, 3, 3)
        num_clases = max(v, int(np.max(y_real)) + 1)
        matriz = np.zeros((num_clases, num_clases), dtype=int)
        for r, p in zip(y_real, y_pred):
            matriz[r, p] += 1
            
        cax = ax3.matshow(matriz, cmap='Blues')
        fig.colorbar(cax, ax=ax3)
        for i in range(num_clases):
            for j in range(num_clases):
                ax3.text(j, i, str(matriz[i, j]), va='center', ha='center', color='black')
        ax3.set_title("Matriz de Confusión")
        ax3.set_xlabel("Predicción")
        ax3.set_ylabel("Real")

    plt.tight_layout()
    plt.show()

def imprimir_hiperparametros(red):
    print("HIPERPARÁMETROS DE LA RED")
    print(f"(u) Neuronas de entrada    : {red.u}")
    print(f"(v) Neuronas de salida     : {red.v}")
    print(f"(L) Capas ocultas          : {red.L}")
    print(f"(b) Neuronas por capa      : {red.b}")
    print(f"(e) Épocas acumuladas      : {red.e}")

def main():
    print("PERCEPTRÓN MULTICAPA (MLP)")
    print("1. Crear nuevo Perceptrón Multicapa")
    print("2. Cargar red desde un archivo CSV")
    opc_inicio = input("Seleccione una opción (1/2): ").strip()

    if opc_inicio == "1":
        u = int(input("Número de neuronas de entrada (u): "))
        v = int(input("Número de neuronas de salida (v): "))
        L = int(input("Número de capas ocultas (L): "))
        b = int(input("Número de neuronas por capa oculta (b): "))
        red = PerceptronMulticapa(u, v, L, b)
    else:
        ruta = input("Ingrese el nombre del archivo CSV de la red (ejemplo: red1.csv): ").strip()
        red = cargar_red(resolver_ruta(ruta))

    imprimir_hiperparametros(red)

    while True:
        print("\nMENÚ")
        print("1. Entrenar la red (Backpropagation)")
        print("2. Probar la red (Feedforward)")
        print("3. Guardar la red en archivo CSV")
        print("4. Salir")
        opc = input("Seleccione una opción (1-4): ").strip()

        if opc == "1":
            ruta_normal = "tarea-2-codigo/assets/"
            archivo = input("Archivo CSV de entrenamiento (ejemplo: train.csv): ").strip()
            ruta_archivo = os.path.join(ruta_normal, archivo)
            try:
                X, y_real, y_onehot = cargar_csv_datos(ruta_archivo, red.u, red.v)
            except Exception as err:
                print(f"[ERROR]: {err}")
                continue

            epocas = int(input("Número de épocas a entrenar: "))
            print("Entrenando red, por favor espere...")
            historial = red.entrenar(X, y_onehot, epocas)

            y_pred = red.predecir(X)
            imprimir_hiperparametros(red)
            graficar_resultados(X, y_real, y_pred, modo='entrenamiento', info_error=historial, v=red.v)

        elif opc == "2":
            ruta_normal = "tarea-2-codigo/assets/"
            archivo = input("Archivo CSV de prueba (ejemplo: test.csv): ").strip()
            ruta_archivo = os.path.join(ruta_normal, archivo)
            try:
                X, y_real, _ = cargar_csv_datos(ruta_archivo, red.u, red.v)
            except Exception as err:
                print(f"[ERROR]: {err}")
                continue

            y_pred = red.predecir(X)
            imprimir_hiperparametros(red)
            graficar_resultados(X, y_real, y_pred, modo='prueba', v=red.v)

        elif opc == "3":
            nombre_salida = input("Ingrese el nombre para guardar el archivo CSV (ejemplo: red1.csv): ").strip()
            ruta_salida = os.path.join(ruta_normal, nombre_salida)
            guardar_red(ruta_salida, red)

        elif opc == "4":
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida.")

main()