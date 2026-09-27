# Nombre del integrante: Diego Goncalves
# Cédula del integrante: 30124687

import matplotlib.pyplot as plt

def activacion_escalon(suma):
    if suma >= 0:
        return 1
    else:
        return 0

def activacion_signo(suma):
    if suma >= 0:
        return 1
    else:
        return -1

def cargar_csv(ruta):
    entradas = []
    esperados = []
    with open(ruta, 'r') as f:
        next(f) # Salta el encabezado
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            valores = [float(v) for v in linea.split(',')]
            entradas.append(valores[:-1])   # Primeras n-1 columnas
            esperados.append(valores[-1])   # Última columna
    return entradas, esperados

def evaluar_perceptron(x, pesos, sesgo, funcion_act):
    # suma = sesgo + (w1*x1 + w2*x2 + ...)
    suma = sesgo
    for i in range(len(x)):
        suma += pesos[i] * x[i]
    return funcion_act(suma)

def graficar(entradas, esperados, predicciones):
    # Toma las primeras dos dimensiones para graficar
    x1 = [p[0] for p in entradas]
    x2 = [p[1] for p in entradas]

    colores_coincidencia = ['green' if esp == pred else 'red' 
                            for esp, pred in zip(esperados, predicciones)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Gráfico 1: Valor Esperado
    axs[0].scatter(x1, x2, c=esperados, cmap='bwr', s=70, edgecolors='k')
    axs[0].set_title("1. Valor Esperado")
    axs[0].set_xlabel("X1"); axs[0].set_ylabel("X2")

    # Gráfico 2: Predicción del Perceptrón
    axs[1].scatter(x1, x2, c=predicciones, cmap='bwr', s=70, edgecolors='k')
    axs[1].set_title("2. Predicción del Perceptrón")
    axs[1].set_xlabel("X1"); axs[1].set_ylabel("X2")

    # Gráfico 3: Coincidencias (Verde: Acierto, Rojo: Fallo)
    axs[2].scatter(x1, x2, c=colores_coincidencia, s=70, edgecolors='k')
    axs[2].set_title("3. Coincidencias (Verde: Acierto, Rojo: Fallo)")
    axs[2].set_xlabel("X1"); axs[2].set_ylabel("X2")

    plt.tight_layout()
    plt.show()

def main():

    ruta_archivo = "C:\\Users\\diego\\Documents\\GitHub\\Aris-2627-1\\tarea-1-codigo\\assets\\"
    archivo_csv = input("Ingrese el nombre del archivo CSV (ej. datos.csv). Asegurese de que el archivo csv esté en el mismo directorio que el archivo de python. Cambie la ruta si es necesario: \n").strip()

    ruta_completa = ruta_archivo + archivo_csv
    entradas, esperados = cargar_csv(ruta_completa)

    num_entradas = len(entradas[0])
    print(f"\nDatos cargados con éxito. Cada muestra tiene {num_entradas} entradas.")

    while True:
        print("\nCONFIGURACIÓN DE PESOS")
        sesgo = float(input("Ingrese el peso del sesgo/bias: "))
        
        pesos = []
        for i in range(num_entradas):
            w = float(input(f"Ingrese el peso para la entrada X{i+1}: "))
            pesos.append(w)

        print("\nSeleccione la función de activación:")
        print("1. Escalón (Salidas: 0 o 1)")
        print("2. Signo   (Salidas: -1 o 1)")
        opcion = input("Opción (1/2): ").strip()
        if opcion == "1":
            funcion_act = activacion_escalon 
        else:
            funcion_act = activacion_signo

        predicciones = [evaluar_perceptron(x, pesos, sesgo, funcion_act) for x in entradas]

        graficar(entradas, esperados, predicciones)

        seguir = input("¿Desea usar otros pesos? (s/n): ").strip().lower()
        if seguir == 'n':
            print("Fin del programa.")
            break

main()