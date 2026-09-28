# Nombre del integrante: Máximo Mujica
# Cédula del integrante: 30727257

# haga su tarea aqui

import matplotlib.pyplot as plt

#Defino x como lista de entradas
#Defino y como salidas esperadas
def cargar_csv(ruta_archivo):
    X = []
    Y = []
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        lineas = f.readlines()
        if not lineas:
            return X, Y

        for linea in lineas[1:]:
            linea = linea.strip()
            if not linea:
                continue
            valores = linea.split(",")
            X.append([float(val) for val in valores[:-1]])
            Y.append(float(valores[-1]))
    return X, Y

def funcion_suma(entradas, pesos, sesgo):
    z = sesgo
    for xi, wi in zip(entradas, pesos):
        z += xi * wi
    return z

def escalon_unitario(z):
    #Funcion de activacion heaviside que vimos en prepa {0,1}
    return 1.0 if z >= 0.0 else 0.0

def funcion_signo(z):
    #Segunda funcion de activación
    return 1.0 if z >= 0.0 else -1.0

def predecir(entradas, pesos, sesgo, func_act):
    z = funcion_suma(entradas, pesos, sesgo)
    return func_act(z)

def mostrar_graficos(X, y_esperado, y_predicho):
    # extraer las dos primeras dimensiones para graficar
    x1 = [fila[0] for fila in X]
    x2 = [fila[1] for fila in X]

    # colores para el tercer gráfico: verde si coinciden, rojo si difieren
    colores_coincidencia = [
        "green" if esp == pred else "red"
        for esp, pred in zip(y_esperado, y_predicho)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Gráfico 1: Valor esperado
    scatter1 = axes[0].scatter(x1, x2, c=y_esperado, cmap="coolwarm", edgecolors="k", alpha=0.8)
    axes[0].set_title("1. Valor Esperado")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.colorbar(scatter1, ax=axes[0])

    # Gráfico 2: Valor predicho
    scatter2 = axes[1].scatter(x1, x2, c=y_predicho, cmap="coolwarm", edgecolors="k", alpha=0.8)
    axes[1].set_title("2. Valor Predicho por el Perceptrón")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    fig.colorbar(scatter2, ax=axes[1])

    # Gráfico 3: Coincidencias (Verde = Coincide, Rojo = Falla)
    axes[2].scatter(x1, x2, c=colores_coincidencia, edgecolors="k", alpha=0.8)
    axes[2].set_title("3. Coincidencia (Verde: Acierto, Rojo: Error)")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")

    # Resumen de métricas
    aciertos = sum(1 for esp, pred in zip(y_esperado, y_predicho) if esp == pred)
    total = len(y_esperado)
    print(f"\n[Resultado] Aciertos exactos: {aciertos}/{total} ({(aciertos/total)*100:.2f}%)")

    plt.tight_layout()
    plt.show()

def main():
    print("========================================")
    print("      SIMULADOR DE PERCEPTRÓN SIMPLE    ")
    print("========================================")

    # Solicitar archivo CSV
    ruta_defecto = "tarea-1-codigo/assets/fuzzy_separables.csv"
    ruta_archivo = input(f"Ingrese la ruta del archivo CSV [Enter para '{ruta_defecto}']: ").strip()
    if not ruta_archivo:
        ruta_archivo = ruta_defecto

    try:
        X, Y = cargar_csv(ruta_archivo)
    except Exception as e:
        print(f"Error al abrir el archivo: {e}")
        return

    if not X:
        print("El archivo no contiene datos válidos.")
        return

    num_entradas = len(X[0])
    print(f"\nArchivo cargado con éxito. Número de entradas (n-1): {num_entradas}")

    # Bucle interactivo para probar pesos
    while True:
        print("\n--- Configuración de parámetros ---")
        try:
            sesgo = float(input("Ingrese el valor del sesgo (bias): "))
            
            pesos = []
            for i in range(num_entradas):
                w = float(input(f"Ingrese el peso w{i+1}: "))
                pesos.append(w)

            print("\nSeleccione la función de activación:")
            print("1. Escalón Unitario (Heaviside): Salidas {0, 1}")
            print("2. Signo (Bipolar): Salidas {-1, 1}")
            opcion_act = input("Opción (1/2): ").strip()

            if opcion_act == "2":
                func_act = funcion_signo
            else:
                func_act = escalon_unitario

        except ValueError:
            print("Entrada inválida. Debe ingresar valores numéricos.")
            continue

        y_pred = [predecir(fila, pesos, sesgo, func_act) for fila in X]

        mostrar_graficos(X, Y, y_pred)

        repetir = input("\n¿Desea probar con otros pesos? (s/n): ").strip().lower()
        if repetir != "s":
            print("Finalizando programa.")
            break


if __name__ == "__main__":
    main()