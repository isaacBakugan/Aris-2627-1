# Nombre del integrante: Veronica Girón
# Cédula del integrante: 29908725

# haga su tarea aqui
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# fijamos la tasa de aprendizaje
ETA = 0.1
    
def cargar_csv(archivo):
    with open(archivo, encoding="utf-8-sig") as f:
        lineas = [linea.strip() for linea in f if linea.strip()]
    
    datos = []
    
    for linea in lineas[1:]:
        partes = linea.split(",")
        fila = [float(valor) for valor in partes]
        datos.append(fila)
        
    # convertimos la lista en una matriz de numpy
    matriz_datos = np.array(datos)
    return matriz_datos
    
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


def activacion_sigmoide(z):
    # f(z) = 1 / (1 + e^-z)
    exponencial = np.exp(-z)
    resultado = 1.0 / (1.0 + exponencial)
    return resultado

def derivada_sigmoide(a):
    # f'(z) = a *(1-a) (a es el resultado de la sigmoide)
    derivada = a * (1.0 - a) 
    return derivada

def crear_red(u, v, L, b):
    pesos = []
    sesgos = []
    
    # Capa oculta 1: recibe u entradas y tiene b neuronas. matriz tamano bxu
    # Matriz de pesos de tamano (b, u)
    pesos_oculta1 = np.random.uniform(-1, 1, size=(b,u)) 
    sesgos_oculta1 = np.random.uniform(-1, 1, size=(b,))
    pesos.append(pesos_oculta1)
    sesgos.append(sesgos_oculta1)
    
    # Capas ocultas intermedias (en caso de que L sea mayor a 1)
    # Reciben "b" entradas (capa anterior) y tienen "b" neuronas. Matriz tamano bxb
    for _ in range(L - 1):
        pesos_capa = np.random.uniform(-1, 1, size=(b,b))
        sesgos_capa = np.random.uniform(-1, 1, size=(b,))
        pesos.append(pesos_capa)
        sesgos.append(sesgos_capa)
    
    # Capa de salida: recibe "b" entradas y tiene "v" neuronas
    pesos_salida = np.random.uniform(-1, 1, size=(v, b))
    sesgos_salida = np.random.uniform(-1, 1, size=(v,))
    pesos.append(pesos_salida)
    sesgos.append(sesgos_salida)
    
    # Guardamos en un diccionario
    red = {
        "u": int(u),
        "v": int(v),
        "L": int(L),
        "b": int(b),
        "epocas": 0, # contador de epocas
        "pesos": pesos,
        "sesgos": sesgos,
    }
    return red

# pasa una muestra de entrada por todas las capas de la red
def forward_pass(red, entrada): # use spanglish sorry
    # Guardamos las activaciones de cada capa. la primera es la entrada
    activaciones = [entrada]
    a_actual = entrada
    
    lista_pesos = red["pesos"]
    lista_sesgos = red["sesgos"]
    
    #Pasamos el dato por cada capa de la red
    for W, s in zip(lista_pesos, lista_sesgos):
        # z = W * a + s (suma ponderada)
        suma_ponderada = np.dot(W, a_actual)
        z = suma_ponderada + s
        
        # aqui llamamos a la funcion de activacion
        a_actual = activacion_sigmoide(z)
        
        # Guardamos el resultado para usarlo luego
        activaciones.append(a_actual)
        
    return activaciones

# La red aprende con este algoritmo: 
# 1. mide el error en la salida
# 2. calcula que tanta culpa (las derivadas del error con respecto a la suma poderada) tiene cada neurona
# 3. ajusta los pesos y sesgos en la direccion opuesta al error
def back_propagation(red, activaciones, objetivo):
    pesos = red["pesos"]
    sesgos = red["sesgos"]
    cant_capas = len(pesos)
    
    deltas = [None] * cant_capas
    
    # Paso 1: calculamos el delta (culpa) en la capa de salida
    salida_obtenida = activaciones[-1] # la salida de la red (o)

    error_salida = salida_obtenida - objetivo # (o - t)
    # Multiplicamos por la derivada : delta = (salida - objetivo) * derivada(salida)
    derivada_salida = derivada_sigmoide(salida_obtenida) # o * (1 - o)
    
    delta_salida = error_salida * derivada_salida
    deltas[-1] = delta_salida
    
    # Paso 2: repartir la "culpa" hacia atras 
    indice_ultima_oculta = cant_capas - 2
    
    # El bucle va hacia atrás: desde la última capa oculta hasta la primera (índice 0)
    for k in range(indice_ultima_oculta, -1, -1):
        # 1. Obtenemos las activaciones de la capa oculta 'k'
        a_capa_k = activaciones[k + 1]
       
        # 2. Obtenemos los pesos y el delta de la capa SIGUIENTE (la que está más cerca de la salida)       
        pesos_siguientes = pesos[k + 1]
        delta_siguiente = deltas[k + 1]
        
        # 3. PROPAGACIÓN: Multiplicamos el delta de la capa siguiente por sus pesos traspuestos.
        # Esto distribuye el error hacia atrás para saber cuánta responsabilidad tuvo cada neurona de la capa k.        
        error_propagado = np.dot(pesos_siguientes.T, delta_siguiente)
        
        # 4. Derivada de la sigmoide de la capa oculta actual: a * (1 - a)
        derivada_capa = derivada_sigmoide(a_capa_k)
       
        # 5. Calculamos el delta de la capa oculta k     
        delta_k = error_propagado * derivada_capa

        # 6. Guardamos el delta en la lista de deltas
        deltas[k] = delta_k
        
    # Paso 3: actualizamos los pesos y sesgos con el descenso del gradiente negativo
    for k in range(cant_capas):
       a_anterior = activaciones[k]
       delta_k = deltas[k]
       
       # calculo del gradiente negativo respecto a los pesos
       matriz_cambio = np.outer(delta_k, a_anterior)
       
       # aplicamos el gradiente negativo
       # Restamos (ETA * matriz_cambio) a los pesos para mover la red en dirección opuesta
       # al incremento del error, reduciendo el error global.
       pesos[k] = pesos[k] - (ETA * matriz_cambio)
       sesgos[k] = sesgos[k] - (ETA * delta_k)
        
# entrenamiento por epocas
def entrenar_red(red, entradas, objetivos_onehot, epocas):
    # Ejecuta el ciclo completo de entrenamiento durante N épocas.
    errores_por_epoca = []
    cant_muestras = len(entradas)

    for _ in range(epocas):
        # Desordenamos aleatoriamente el orden de los patrones en cada época.
        # Esto evita que la red aprenda la secuencia fija de las filas del CSV.
        indices = np.random.permutation(cant_muestras)
        error_acumulado_epoca = 0.0

        for idx in indices:
            x = entradas[idx]
            t = objetivos_onehot[idx]

            # 1. forward pass
            activaciones = forward_pass(red, x)
            salida = activaciones[-1]

            # 2. Medición del Error Cuadrático Medio del patrón: E = 0.5 * sum((t - o)^2)
            diferencia = t - salida
            diferencia_cuadrado = diferencia ** 2
            error_muestra = 0.5 * np.sum(diferencia_cuadrado)
            error_acumulado_epoca = error_acumulado_epoca + error_muestra

            # 3. Corrección de pesos inmediata mediante Retropropagación
            back_propagation(red, activaciones, t)

        # Calculamos el error promedio cuadrático de toda la época
        error_promedio = error_acumulado_epoca / cant_muestras
        errores_por_epoca.append(error_promedio)

    # Actualizamos el contador global de épocas entrenadas
    red["epocas"] = red["epocas"] + epocas
    return errores_por_epoca


def preparar_clases(datos):
    entradas = datos[:, :-1]  # Todas las columnas excepto la última
    clases_raw = datos[:, -1] # Únicamente la última columna (clases reales)

    # Obtenemos un array ordenado con los valores únicos de clases presentes en el dataset
    clases_unicas = np.unique(clases_raw)
    clases_unicas.sort()

    v = len(clases_unicas)
    cant_muestras = len(clases_raw)

    # Creamos la matriz de respuesta esperada inicializada en ceros
    objetivos_onehot = np.zeros((cant_muestras, v))

    for i in range(cant_muestras):
        clase_actual = clases_raw[i]
        # Buscamos en qué índice (0, 1, ..., v-1) está esa clase
        posicion = np.where(clases_unicas == clase_actual)[0][0]
        # Asignamos un 1 en la neurona que representa a esa clase
        objetivos_onehot[i, posicion] = 1.0

    return entradas, objetivos_onehot, clases_unicas, clases_raw

def evaluar(red, entradas, clases_raw, clases_unicas):
    # Realiza una evaluación (Forward únicamente) para medir aciertos y construir la Matriz de Confusión.
    aciertos = []
    predicciones = []
    v = red["v"]

    # La Matriz de Confusión es una tabla de dimensiones vxv
    # Filas: Clase Real | Columnas: Clase Predicha por la Red
    matriz_confusion = np.zeros((v, v), dtype=int)
    cant_muestras = len(entradas)

    for i in range(cant_muestras):
        x = entradas[i]
        clase_real = clases_raw[i]

        # forward pass
        activaciones = forward_pass(red, x)
        salida_red = activaciones[-1]

        # La neurona de salida con mayor valor de activación (np.argmax) representa la decisión de la red
        pos_predicha = np.argmax(salida_red)
        clase_predicha = clases_unicas[pos_predicha]

        predicciones.append(clase_predicha)

        # Verificamos si la predicción coincide con la clase real
        if clase_predicha == clase_real:
            aciertos.append(True)
        else:
            aciertos.append(False)

        # Contabilizamos la combinación real/predicha en la matriz de confusión
        pos_real = np.where(clases_unicas == clase_real)[0][0]
        matriz_confusion[pos_real, pos_predicha] = (
            matriz_confusion[pos_real, pos_predicha] + 1
        )

    return predicciones, aciertos, matriz_confusion
        

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

def graficar_resultados(entradas, clases_raw, aciertos, dato_grafico3, es_entrenamiento):
  
    num_entradas = entradas.shape[1]
    fig = plt.figure(figsize=(15, 4))

   
    if num_entradas >= 3:
        # Si el CSV tiene 3 o más entradas, usamos una proyección tridimensional
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.scatter(
            entradas[:, 0],
            entradas[:, 1],
            entradas[:, 2],
            c=clases_raw,
            cmap="coolwarm",
        )
        ax1.set_zlabel("x3")
    else:
        # Si el CSV es de 2 entradas, usamos un plano 2D 
        ax1 = fig.add_subplot(1, 3, 1)
        if num_entradas > 1:
            x2 = entradas[:, 1]
        else:
            x2 = np.zeros(len(entradas))

        ax1.scatter(entradas[:, 0], x2, c=clases_raw, cmap="coolwarm")
        ax1.set_ylabel("x2")

    ax1.set_title("Valor esperado")
    ax1.set_xlabel("x1")

    # Coincidencia (Verde si la red acertó, Rojo si falló)
    colores = []
    for acierto in aciertos:
        if acierto:
            colores.append("green")
        else:
            colores.append("red")

    if num_entradas >= 3:
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.scatter(
            entradas[:, 0], entradas[:, 1], entradas[:, 2], c=colores
        )
        ax2.set_zlabel("x3")
    else:
        ax2 = fig.add_subplot(1, 3, 2)
        if num_entradas > 1:
            x2 = entradas[:, 1]
        else:
            x2 = np.zeros(len(entradas))

        ax2.scatter(entradas[:, 0], x2, c=colores)
        ax2.set_ylabel("x2")

    ax2.set_title("Coincidencia")
    ax2.set_xlabel("x1")

    # Curva de Error O Matriz de Confusión
    ax3 = fig.add_subplot(1, 3, 3)

    if es_entrenamiento:
        # Caso Entrenamiento: Graficamos la disminución del error por cada época
        lista_errores = dato_grafico3
        ax3.plot(lista_errores, color="blue")
        ax3.set_title("Error de la red")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("Error promedio")
    else:
        # Caso Prueba: Dibujamos un mapa de calor (matshow) de la matriz de confusión
        matriz = dato_grafico3
        ax3.matshow(matriz, cmap="Blues")
        ax3.set_title("Matriz de confusión")
        ax3.set_xlabel("predicha")
        ax3.set_ylabel("real")

        # Anotamos los valores numéricos dentro de cada casilla del mapa de calor
        filas = matriz.shape[0]
        columnas = matriz.shape[1]
        for f in range(filas):
            for c in range(columnas):
                valor = matriz[f, c]
                ax3.text(c, f, str(valor), va="center", ha="center")

    plt.tight_layout()
    plt.show()
    

# guarda y carga la red en un csv
def guardar_red(red, nombre_archivo):
    u = red["u"]
    v = red["v"]
    L = red["L"]
    b = red["b"]
    epocas = red["epocas"]

    # c = max(u, b)
    if u > b:
        c = u
    else:
        c = b

    num_columnas_pesos = c + 1  # w0 (sesgo) + c columnas para pesos de entrada

    encabezado = ["tipo", "capa", "posicion"]
    for i in range(num_columnas_pesos):
        encabezado.append(f"w{i}")

    filas = [encabezado]

    # Escribimos los hiperparámetros en las primeras 5 filas
    relleno = [""] * (len(encabezado) - 2)
    filas.append(["u", str(u)] + relleno)
    filas.append(["v", str(v)] + relleno)
    filas.append(["L", str(L)] + relleno)
    filas.append(["b", str(b)] + relleno)
    filas.append(["e", str(epocas)] + relleno)

    # Escribimos los parámetros de cada neurona
    lista_pesos = red["pesos"]
    lista_sesgos = red["sesgos"]
    cant_capas = len(lista_pesos)

    for i in range(cant_capas):
        W = lista_pesos[i]
        s = lista_sesgos[i]

        # 'h' para capas ocultas, 'o' para capa de salida
        if i < cant_capas - 1:
            tipo = "h"
        else:
            tipo = "o"

        capa_num = i + 1  # La numeración de capas empieza en 1
        num_neuronas = W.shape[0]

        for pos in range(num_neuronas):
            fila = [tipo, str(capa_num), str(pos)]

            # La cuarta columna (w0) almacena el sesgo
            sesgo_val = str(s[pos])
            fila.append(sesgo_val)

            # Las columnas subsiguientes (w1, w2) almacenan los pesos
            pesos_neurona = W[pos]
            for peso in pesos_neurona:
                fila.append(str(peso))

            # Rellenamos con celdas vacías si la neurona tiene menos pesos que c
            faltantes = len(encabezado) - len(fila)
            if faltantes > 0:
                fila.extend([""] * faltantes)

            filas.append(fila)

    # Guardamos en el archivo CSV
    with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerows(filas)

    print("Red guardada exitosamente.")

# Lee un archivo CSV guardado previamente y reconstruye las matrices de pesos, sesgos e hiperparámetros de la red.
def cargar_red(nombre_archivo):
    # 1. Asegurar la extensión .csv
    if not nombre_archivo.lower().endswith(".csv"):
        nombre_archivo_ext = nombre_archivo + ".csv"
    else:
        nombre_archivo_ext = nombre_archivo

    # 2. Rutas alternativas donde buscará el archivo si no está en la raíz
    rutas_posibles = [
        nombre_archivo_ext,
        os.path.join("tarea-2-codigo", nombre_archivo_ext),
        os.path.join("tarea-2-codigo", "assets", nombre_archivo_ext),
        os.path.join("assets", nombre_archivo_ext),
    ]

    ruta_encontrada = None
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            ruta_encontrada = ruta
            break

    if ruta_encontrada is None:
        raise FileNotFoundError(f"No se encontró el archivo '{nombre_archivo_ext}' en ninguna de las carpetas conocidas.")

    # 3. Leer el archivo encontrado
    with open(ruta_encontrada, encoding="utf-8-sig") as f:
        lector = csv.reader(f)
        lineas = [fila for fila in lector if fila]

    # Leer hiperparámetros
    u = int(lineas[1][1])
    v = int(lineas[2][1])
    L = int(lineas[3][1])
    b = int(lineas[4][1])
    epocas = int(lineas[5][1])

    pesos = []
    sesgos = []

    pesos.append(np.zeros((b, u)))
    sesgos.append(np.zeros(b))

    for _ in range(L - 1):
        pesos.append(np.zeros((b, b)))
        sesgos.append(np.zeros(b))

    pesos.append(np.zeros((v, b)))
    sesgos.append(np.zeros(v))

    filas_neuronas = lineas[6:]
    for fila in filas_neuronas:
        if not fila:
            continue

        capa_num = int(fila[1]) - 1
        pos = int(fila[2])

        sesgo_val = float(fila[3])
        sesgos[capa_num][pos] = sesgo_val

        pesos_texto = fila[4:]
        pesos_numeros = [float(val) for val in pesos_texto if val != ""]

        cant_pesos = len(pesos_numeros)
        pesos[capa_num][pos, :cant_pesos] = pesos_numeros

    red = {
        "u": u,
        "v": v,
        "L": L,
        "b": b,
        "epocas": epocas,
        "pesos": pesos,
        "sesgos": sesgos,
    }

    print(f"Red cargada exitosamente desde '{ruta_encontrada}'.")
    return red

# Verifica que las dimensiones del CSV cargado coincidan con u y v de la red.
def validar_red(red, entradas, clases_unicas):
    u_datos = entradas.shape[1]
    v_datos = len(clases_unicas)

    if red["u"] != u_datos or red["v"] != v_datos:
        print("\n" + "=" * 55)
        print("ERROR: La red no coincide con el archivo de datos.")
        print(f"La red espera u={red['u']} entradas y v={red['v']} salidas.")
        print(f"El archivo tiene {u_datos} entradas y {v_datos} clases.")
        print(f"Crea o carga una red con u={u_datos} y v={v_datos}.")
        print("=" * 55 + "\n")
        return False
    return True

# Muestra los hiperparámetros actuales e imprime el total acumulado de épocas
def imprimir_hiperparametros(red):
    print("\n--- Hiperparámetros ---")
    print("u (entradas):", red["u"])
    print("v (salidas):", red["v"])
    print("L (capas ocultas):", red["L"])
    print("b (neuronas por capa):", red["b"])
    print("Épocas acumuladas:", red["epocas"])
    print("Tasa de aprendizaje (eta):", ETA)
    print("-----------------------\n")


def main():
    red = None

    print("=== PERCEPTRÓN MULTICAPA ===")
    print("1. Crear red nueva")
    print("2. Cargar red desde archivo")

    # Selección inicial: Crear o Cargar
    while True:
        opcion_inicio = input("Elija una opción (1 o 2): ").strip()

        if opcion_inicio == "1":
            u = int(leer_numero("Número de neuronas de entrada (u): "))
            v = int(leer_numero("Número de neuronas de salida (v): "))
            L = int(leer_numero("Número de capas ocultas (L): "))
            b = int(leer_numero("Número de neuronas por capa (b): "))
            red = crear_red(u, v, L, b)
            break
        elif opcion_inicio == "2":
            archivo_red = input("Nombre del archivo de la red: ").strip()
            try:
                red = cargar_red(archivo_red)
                break
            except Exception as e:
                print("Error al cargar el archivo de la red:", e)
        else:
            print("Opción no válida, intente otra vez.")

    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Entrenar")
        print("2. Probar")
        print("3. Guardar red")
        print("4. Salir")

        opcion = input("Elija una opción (1-4): ").strip()

        if opcion == "1":
            archivo = input("Ingrese la ruta del csv de entrenamiento: ").strip()
            try:
                datos = cargar_csv(archivo)
            except FileNotFoundError:
                print("No se encontró el archivo.")
                continue

            entradas, objetivos_onehot, clases_unicas, clases_raw = preparar_clases(datos)

            # Validar compatibilidad entre el dataset y la red
            if not validar_red(red, entradas, clases_unicas):
                continue

            epocas = int(leer_numero("Número de épocas a entrenar: "))
            errores = entrenar_red(red, entradas, objetivos_onehot, epocas)

            _, aciertos, _ = evaluar(red, entradas, clases_raw, clases_unicas)

            imprimir_hiperparametros(red)
            print("Aciertos:", sum(aciertos), "de", len(entradas))

            graficar_resultados(
                entradas, clases_raw, aciertos, errores, es_entrenamiento=True
            )

        elif opcion == "2":
            archivo = input("Ingrese la ruta del csv de prueba: ").strip()
            try:
                datos = cargar_csv(archivo)
            except FileNotFoundError:
                print("No se encontró el archivo.")
                continue

            entradas, _, clases_unicas, clases_raw = preparar_clases(datos)

            # Validar compatibilidad entre el dataset y la red
            if not validar_red(red, entradas, clases_unicas):
                continue

            _, aciertos, matriz = evaluar(red, entradas, clases_raw, clases_unicas)

            imprimir_hiperparametros(red)
            print("Aciertos:", sum(aciertos), "de", len(entradas))

            graficar_resultados(
                entradas, clases_raw, aciertos, matriz, es_entrenamiento=False
            )

        elif opcion == "3":
            nombre_guardar = input("Nombre del archivo para guardar (.csv): ").strip()
            try:
                guardar_red(red, nombre_guardar)
            except Exception as e:
                print("Error al guardar la red:", e)

        elif opcion == "4":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida, intente otra vez.")


if __name__ == "__main__":
    main()