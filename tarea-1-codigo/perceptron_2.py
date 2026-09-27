# Nombre del integrante: Verónica Girón 
# Cédula del integrante: 29908725

# haga su tarea aqui
import matplotlib.pyplot as plt
    
    
def cargar_csv(archivo):
    with open(archivo, encoding="utf-8-sig") as f:
        lineas = [linea.strip() for linea in f if linea.strip()]
    
    datos = []
    for linea in lineas[1:]:
        partes = linea.split(",")
        fila = [float(valor) for valor in partes]
        datos.append(fila)
    return datos

def leer_numero(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("Eso no es un numero, intenta otra vez.")

def funcion_suma_ponderada(entradas, pesos, sesgo):
    total = sesgo
    for entrada, peso in zip(entradas, pesos):
        total += entrada * peso
    return total

def activacion_escalon(z): 
    if z > 0:
        return 1
    return 0

def clasificar(valor):
    if valor >= 0.5:
        return 1
    return 0

def activacion_signo(z):
    if z >= 0:
        return 1
    return -1

def escojer_pesos(cantidad_entradas):
    sesgo = leer_numero("Bias (b): ")
    pesos = []
    
    for indice in range(cantidad_entradas):
        peso = leer_numero(f"Peso x{indice + 1}: ")
        pesos.append(peso)
        
    return sesgo, pesos
    
def escoger_funactivacion():
    print("Funciones de activacion disponibles:")
    print("1. Escalon")
    print("2. Signo")
    
    while True:
        opcion = input("Elige una opcion(1 o 2): ").strip()
        if opcion == "1":
            return activacion_escalon, "Escalon"
        if opcion == "2":
            return activacion_signo, "Signo"
        print("Opcion no valida.")

def predecir_fila(entradas, pesos, sesgo, activacion):
    z = funcion_suma_ponderada(entradas, pesos, sesgo)
    return activacion(z) 

def ajustar_prediccion(predeccion, nombre_activacion, etiquetas):
    # si el csv usa 0 y 1, el -1 de signo cuenta como 0
    if nombre_activacion == "Signo" and -1 not in etiquetas and predeccion == -1:
        return 0
    # si el csv usa -1 y 1, el 0 de escalon cuenta como -1
    if nombre_activacion == "Escalon" and 0 not in etiquetas and predeccion == 0:
        return -1
    return predeccion
        


def graficar_resultados(datos, predecciones, aciertos):
    x1 = [fila[0] for fila in datos]
    
    if len(datos[0]) > 2:
        x2 = [fila[1] for fila in datos]
    else:
        x2 = [0 for _ in datos]
    
    esperados = [fila[-1] for fila in datos]
    colores = ["green" if acierto else "red" for acierto in aciertos]

    figura, ejes = plt.subplots(1, 3, figsize=(15, 4))

    ejes[0].scatter(x1, x2, c=esperados, cmap="coolwarm")
    ejes[0].set_title("Valor esperado")
    ejes[0].set_xlabel("x1")
    ejes[0].set_ylabel("x2")

    ejes[1].scatter(x1, x2, c=predecciones, cmap="coolwarm")
    ejes[1].set_title("Valor predicho")
    ejes[1].set_xlabel("x1")
    ejes[1].set_ylabel("x2")

    ejes[2].scatter(x1, x2, c=colores)
    ejes[2].set_title("Coincidencia")
    ejes[2].set_xlabel("x1")
    ejes[2].set_ylabel("x2")
    plt.tight_layout()
    plt.show()
        
def main():
    archivo = input("Ingrese la ruta del csv: ").strip()

    try:
        datos = cargar_csv(archivo)
    except FileNotFoundError:
        print("No se encontro el archivo.")
        return

    if not datos:
        print("El archivo no tiene datos.")
        return

    cantidad_entradas = len(datos[0]) - 1
    etiquetas = [fila[-1] for fila in datos]

    while True:
        sesgo, pesos = escojer_pesos(cantidad_entradas)
        activacion, nombre_activacion = escoger_funactivacion()

        predecciones = []
        aciertos = []
        
        for fila in datos:
            entradas = fila[:-1]
            esperado = fila[-1]
            predeccion = predecir_fila(entradas, pesos, sesgo, activacion)
            predeccion = ajustar_prediccion(predeccion, nombre_activacion, etiquetas)
            predecciones.append(predeccion)
            aciertos.append(clasificar(predeccion) == clasificar(esperado))

        print("Activación usada:", nombre_activacion)
        print("Aciertos:", sum(aciertos), "de", len(aciertos))

        graficar_resultados(datos, predecciones, aciertos)

        repetir = input("¿Quieres probar otros pesos? (s/n): ").strip().lower()
        if repetir != "s":
            break

if __name__ == "__main__":
    main()