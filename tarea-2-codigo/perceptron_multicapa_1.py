# Nombre del integrante: Máximo Mujica
# Cédula del integrante: 30727257
"""
Tarea 2 — Perceptrón multicapa desde cero (solo NumPy + Matplotlib).

Organización del archivo (cada sección es independiente de las siguientes):

    1. CAPA MATEMÁTICA   -> PerceptronMulticapa (forward, backprop, gradientes).
    2. CAPA DE DATOS     -> lectura de CSV de datos (convención de la Tarea 1) y validación.
    3. PERSISTENCIA      -> guardar_red / cargar_red (formato CSV u, v, L, b, e + neuronas).
    4. CAPA VISUAL       -> los tres gráficos (matplotlib, con proyección 3D).
    5. CLI               -> menús interactivos por consola.

Convenciones:
    * Hay v neuronas de salida = v clases. Las etiquetas del CSV (p. ej. -1/1 o 1..4)
      se mapean a índices 0..v-1 ordenándolas de menor a mayor, y la salida esperada
      es un vector one-hot.
    * Capas ocultas: tangente hiperbólica. Capa de salida: sigmoide. Error: MSE.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proyección '3d')


# =============================================================================
# 1. CAPA MATEMÁTICA
# =============================================================================

def tanh(z: np.ndarray) -> np.ndarray:
    """Tangente hiperbólica: (e^z - e^-z) / (e^z + e^-z), salida en (-1, 1)."""
    ez = np.exp(np.clip(z, -50.0, 50.0))
    emz = 1.0 / ez
    return (ez - emz) / (ez + emz)


def tanh_derivada(a: np.ndarray) -> np.ndarray:
    """Derivada de tanh expresada en función de su SALIDA a = tanh(z): 1 - a²."""
    return 1.0 - a * a


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide: 1 / (1 + e^-z), salida en (0, 1). El clip evita overflow en exp."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def sigmoide_derivada(a: np.ndarray) -> np.ndarray:
    """Derivada de la sigmoide en función de su SALIDA a = sigmoide(z): a(1 - a)."""
    return a * (1.0 - a)


class PerceptronMulticapa:
    """
    Red feedforward totalmente conectada.

    Representación interna (la clave para entender todo el código):
        tamanos = [u, h1, h2, ..., hL, v]      # neuronas por capa, incluida la entrada
        pesos[k]  : matriz (tamanos[k], tamanos[k+1])  -> columna j = pesos de la neurona j de la capa k+1
        sesgos[k] : vector (tamanos[k+1],)             -> un bias por neurona de la capa k+1
    Así, para un lote X de forma (N, u):   Z = A_prev @ pesos[k] + sesgos[k]
    """

    def __init__(
        self,
        n_entradas: int,
        n_salidas: int,
        neuronas_ocultas: Sequence[int],
        tasa_aprendizaje: float = 0.1,
        semilla: Optional[int] = None,
    ) -> None:
        self.u: int = n_entradas
        self.v: int = n_salidas
        self.capas_ocultas: List[int] = list(neuronas_ocultas)
        self.tamanos: List[int] = [n_entradas, *self.capas_ocultas, n_salidas]
        self.tasa_aprendizaje: float = tasa_aprendizaje
        self.epocas: int = 0                       # épocas ACUMULADAS (se guarda como 'e')
        self.clases: Optional[np.ndarray] = None   # etiquetas ordenadas; índice = neurona de salida
        self._rng = np.random.default_rng(semilla)

        self.pesos: List[np.ndarray] = []
        self.sesgos: List[np.ndarray] = []
        for n_in, n_out in zip(self.tamanos[:-1], self.tamanos[1:]):
            # Inicialización Xavier/Glorot uniforme: mantiene la varianza de las señales estable.
            limite = np.sqrt(6.0 / (n_in + n_out))
            self.pesos.append(self._rng.uniform(-limite, limite, size=(n_in, n_out)))
            self.sesgos.append(np.zeros(n_out))

    # ---------------------------------------------------------------- clases
    def ajustar_clases(self, y: np.ndarray) -> None:
        """Define el mapeo etiqueta -> neurona de salida (etiquetas únicas, ordenadas)."""
        self.clases = np.unique(y)

    def codificar(self, y: np.ndarray) -> np.ndarray:
        """Etiquetas originales -> índices 0..v-1. Falla si hay etiquetas desconocidas."""
        assert self.clases is not None, "Las clases no han sido definidas."
        idx = np.clip(np.searchsorted(self.clases, y), 0, len(self.clases) - 1)
        if not np.all(self.clases[idx] == y):
            raise ValueError("Hay etiquetas que no pertenecen a las clases de la red.")
        return idx

    # ----------------------------------------------------------- forward pass
    def adelante(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Alimentación hacia delante. Devuelve TODAS las activaciones [A0, A1, ..., A_{L+1}]:
            A0 = X                      (N, u)
            A_k = f(A_{k-1} @ W_k + b_k) (N, tamanos[k])
        Las intermedias se necesitan después en la retropropagación.
        """
        activaciones = [X]
        ultima = len(self.pesos) - 1
        for k, (W, b) in enumerate(zip(self.pesos, self.sesgos)):
            Z = activaciones[-1] @ W + b                   # (N, n_in) @ (n_in, n_out) + (n_out,)
            activaciones.append(sigmoide(Z) if k == ultima else tanh(Z))
        return activaciones

    def salida(self, X: np.ndarray) -> np.ndarray:
        """Salida de la red (N, v)."""
        return self.adelante(X)[-1]

    def predecir_indices(self, X: np.ndarray) -> np.ndarray:
        """Neurona de salida con mayor activación para cada vector (argmax)."""
        return np.argmax(self.salida(X), axis=1)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predicción en las etiquetas originales del CSV."""
        assert self.clases is not None, "Las clases no han sido definidas."
        return self.clases[self.predecir_indices(X)]

    # ---------------------------------------------------------- backprop
    def retropropagar(
        self, activaciones: List[np.ndarray], Y: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Retropropagación del error. Minimiza  E = 1/(2N) · Σ ||A_salida - Y||².

        delta_k = dE/dZ_k (error "local" de la capa k, forma (N, n_k)):
            capa de salida:  delta = (A_out - Y) ⊙ f'(A_out) / N
            capa oculta:     delta = (delta_siguiente @ W_siguienteᵀ) ⊙ f'(A_k)
        Gradientes de la capa k:
            dW_k = A_{k-1}ᵀ @ delta_k      (n_in, n_out)  <- misma forma que pesos[k]
            db_k = Σ_filas delta_k         (n_out,)
        Se calculan todos los gradientes ANTES de actualizar, con los pesos viejos.
        """
        n_capas = len(self.pesos)
        grad_W: List[np.ndarray] = [np.empty(0)] * n_capas
        grad_b: List[np.ndarray] = [np.empty(0)] * n_capas
        N = Y.shape[0]

        A_out = activaciones[-1]
        delta = (A_out - Y) * sigmoide_derivada(A_out) / N
        for k in reversed(range(n_capas)):
            grad_W[k] = activaciones[k].T @ delta
            grad_b[k] = delta.sum(axis=0)
            if k > 0:
                # Propaga el error hacia la capa anterior (activaciones[k] es su salida tanh).
                delta = (delta @ self.pesos[k].T) * tanh_derivada(activaciones[k])
        return grad_W, grad_b

    def error_mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Error cuadrático medio sobre todo el conjunto (Y es one-hot)."""
        return float(np.mean((self.salida(X) - Y) ** 2))

    def entrenar(
        self, X: np.ndarray, y: np.ndarray, epocas: int, tam_lote: int = 32
    ) -> List[float]:
        """
        Descenso por gradiente con mini-lotes. Devuelve el MSE al final de cada época
        y suma las épocas realizadas a self.epocas (entrenamiento incremental).
        """
        if self.clases is None:
            self.ajustar_clases(y)
        Y = np.eye(self.v)[self.codificar(y)]          # one-hot (N, v)
        N = X.shape[0]
        historial: List[float] = []
        for _ in range(epocas):
            orden = self._rng.permutation(N)           # barajar en cada época
            for ini in range(0, N, tam_lote):
                lote = orden[ini:ini + tam_lote]
                grad_W, grad_b = self.retropropagar(self.adelante(X[lote]), Y[lote])
                for k in range(len(self.pesos)):
                    self.pesos[k] -= self.tasa_aprendizaje * grad_W[k]
                    self.sesgos[k] -= self.tasa_aprendizaje * grad_b[k]
            historial.append(self.error_mse(X, Y))
            self.epocas += 1
        return historial

    def matriz_confusion(self, y_real_idx: np.ndarray, y_pred_idx: np.ndarray) -> np.ndarray:
        """Filas = clase real, columnas = clase predicha."""
        cm = np.zeros((self.v, self.v), dtype=int)
        np.add.at(cm, (y_real_idx, y_pred_idx), 1)
        return cm


# =============================================================================
# 2. CAPA DE DATOS
# =============================================================================

def cargar_datos(ruta: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee un CSV con encabezado (convención de la Tarea 1): las primeras n-1 columnas
    son las entradas y la n-ésima es la clase. Devuelve X (N, n-1) e y (N,).
    """
    X: List[List[float]] = []
    Y: List[float] = []
    with open(ruta, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    for linea in lineas[1:]:                           # lineas[0] es el encabezado
        linea = linea.strip()
        if not linea:
            continue
        valores = linea.split(",")
        X.append([float(v) for v in valores[:-1]])
        Y.append(float(valores[-1]))
    if not X:
        raise ValueError("El archivo no contiene datos válidos.")
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def validar_compatibilidad(red: PerceptronMulticapa, X: np.ndarray, y: np.ndarray) -> Optional[str]:
    """Devuelve None si el dataset encaja con la red; si no, un mensaje con la configuración esperada."""
    n_clases = len(np.unique(y))
    if X.shape[1] != red.u or (red.clases is None and n_clases != red.v):
        return (
            f"El archivo tiene {X.shape[1]} entradas y {n_clases} clases, pero la red tiene "
            f"u={red.u} entradas y v={red.v} salidas.\n"
            f"Configure la red con u={X.shape[1]} neuronas de entrada y v={n_clases} neuronas de salida."
        )
    if red.clases is not None and not np.all(np.isin(y, red.clases)):
        return (
            f"El archivo contiene etiquetas {np.unique(y).tolist()} y la red conoce "
            f"las clases {red.clases.tolist()}."
        )
    return None


# =============================================================================
# 3. PERSISTENCIA (CSV de la red)
# =============================================================================
# Estructura del archivo (ancho fijo = 3 + c + 1 columnas, con c = max(u, b)):
#
#   fila 0 : encabezado  tipo,capa,indice,w0,...,w{c-1},bias
#   filas 1-5 : hiperparámetros  u,<valor>,,,...   v,...   L,...   b,...   e,...
#   resto  : UNA FILA POR NEURONA
#            tipo ('h' oculta / 'o' salida), capa (ocultas 1..L, salida = L+1),
#            índice dentro de la capa (base 0), pesos de las entradas 0..n_in-1
#            (se rellenan con vacío hasta c) y, en la última columna, el bias.
#   u = entradas, v = salidas, L = capas ocultas, b = neuronas por capa oculta
#   (el máximo si las capas difieren), e = épocas acumuladas.

def _fmt(x: float) -> str:
    return format(float(x), ".17g")                    # 17 cifras: ida y vuelta exacta


def guardar_red(red: PerceptronMulticapa, ruta: Path) -> None:
    L = len(red.capas_ocultas)
    b = max(red.capas_ocultas, default=0)
    c = max(red.u, b)
    ancho = 3 + c + 1

    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "capa", "indice"] + [f"w{i}" for i in range(c)] + ["bias"])
        for letra, valor in (("u", red.u), ("v", red.v), ("L", L), ("b", b), ("e", red.epocas)):
            w.writerow([letra, valor] + [""] * (ancho - 2))   # columnas 3..c+3 vacías

        n_capas = len(red.pesos)
        for k, (W, bias) in enumerate(zip(red.pesos, red.sesgos), start=1):
            tipo = "o" if k == n_capas else "h"
            n_in, n_out = W.shape
            for j in range(n_out):
                pesos_j = [_fmt(x) for x in W[:, j]] + [""] * (c - n_in)
                w.writerow([tipo, k, j, *pesos_j, _fmt(bias[j])])


def cargar_red(ruta: Path) -> PerceptronMulticapa:
    hiper: Dict[str, int] = {}
    neuronas: Dict[int, Dict[int, Tuple[List[str], str]]] = {}
    with open(ruta, "r", newline="", encoding="utf-8") as f:
        for fila in csv.reader(f):
            if not fila or not fila[0].strip():
                continue
            marca = fila[0].strip()
            if marca in ("u", "v", "L", "b", "e"):
                hiper[marca] = int(float(fila[1]))
            elif marca in ("h", "o"):
                capa, idx = int(fila[1]), int(fila[2])
                neuronas.setdefault(capa, {})[idx] = (fila[3:-1], fila[-1])
            # cualquier otra cosa (el encabezado) se ignora

    faltan = [h for h in ("u", "v", "L", "b", "e") if h not in hiper]
    if faltan:
        raise ValueError(f"Faltan hiperparámetros en el archivo: {faltan}")
    u, v, L = hiper["u"], hiper["v"], hiper["L"]
    if sorted(neuronas) != list(range(1, L + 2)):
        raise ValueError(f"Se esperaban las capas 1..{L + 1} y se encontraron {sorted(neuronas)}.")

    # Reconstruir tamaños: n_out de cada capa = cuántas neuronas tiene.
    tamanos_capa = [len(neuronas[k]) for k in range(1, L + 2)]
    if tamanos_capa[-1] != v:
        raise ValueError(f"La capa de salida tiene {tamanos_capa[-1]} neuronas pero v={v}.")

    red = PerceptronMulticapa(u, v, tamanos_capa[:-1])
    entrada = u
    for k in range(1, L + 2):
        n_out = tamanos_capa[k - 1]
        W = np.zeros((entrada, n_out))
        bias = np.zeros(n_out)
        for j in range(n_out):
            celdas, celda_bias = neuronas[k][j]
            W[:, j] = [float(x) for x in celdas[:entrada]]
            bias[j] = float(celda_bias)
        red.pesos[k - 1], red.sesgos[k - 1] = W, bias
        entrada = n_out
    red.epocas = hiper["e"]
    return red


# =============================================================================
# 4. CAPA VISUAL
# =============================================================================

def _scatter(ax, X: np.ndarray, **kw):
    """Scatter 2D (u<=2) o 3D (u>=3, primeras tres características)."""
    if X.shape[1] >= 3:
        return ax.scatter(X[:, 0], X[:, 1], X[:, 2], **kw)
    if X.shape[1] == 2:
        return ax.scatter(X[:, 0], X[:, 1], **kw)
    return ax.scatter(X[:, 0], np.zeros(len(X)), **kw)


def _etiquetar_ejes(ax, u: int) -> None:
    ax.set_xlabel("x1")
    if u >= 2:
        ax.set_ylabel("x2")
    if u >= 3:
        ax.set_zlabel("x3")


def mostrar_resultados(
    red: PerceptronMulticapa,
    X: np.ndarray,
    y: np.ndarray,
    historial: Optional[Sequence[float]] = None,
    titulo: str = "",
) -> None:
    """
    Tres gráficos: (1) datos coloreados por clase real, (2) acierto/error por vector,
    (3) curva de error por época (si se entrenó) o matriz de confusión (si se probó).
    """
    real_idx = red.codificar(y)
    pred_idx = red.predecir_indices(X)
    acierto = real_idx == pred_idx
    proy = "3d" if X.shape[1] >= 3 else None

    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(titulo)

    ax1 = fig.add_subplot(1, 3, 1, projection=proy)
    sc = _scatter(ax1, X, c=real_idx, cmap="viridis", edgecolors="k", alpha=0.8)
    ax1.set_title("1. Datos (color = clase real)")
    _etiquetar_ejes(ax1, X.shape[1])
    fig.colorbar(sc, ax=ax1, shrink=0.6, label="índice de clase")

    ax2 = fig.add_subplot(1, 3, 2, projection=proy)
    if acierto.any():
        _scatter(ax2, X[acierto], c="green", marker="o", alpha=0.7, label="Correcto")
    if (~acierto).any():
        _scatter(ax2, X[~acierto], c="red", marker="x", s=50, label="Incorrecto")
    ax2.set_title(f"2. Predicción ({acierto.sum()}/{len(acierto)} correctas)")
    _etiquetar_ejes(ax2, X.shape[1])
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    if historial is not None:
        ax3.plot(range(1, len(historial) + 1), historial, color="tab:blue")
        ax3.set_title("3. Error cuadrático medio por época")
        ax3.set_xlabel("Época (de esta sesión de entrenamiento)")
        ax3.set_ylabel("MSE")
        ax3.grid(alpha=0.3)
    else:
        cm = red.matriz_confusion(real_idx, pred_idx)
        plt.sca(ax3)
        plt.matshow(cm, fignum=False, cmap="Blues")    # fignum=False -> dibuja en el eje actual
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, str(cm[i, j]), ha="center", va="center")
        etiquetas = [f"{c:g}" for c in red.clases]     # type: ignore[union-attr]
        ax3.set_xticks(range(len(etiquetas)), etiquetas)
        ax3.set_yticks(range(len(etiquetas)), etiquetas)
        ax3.set_xlabel("Clase predicha")
        ax3.set_ylabel("Clase real")
        ax3.set_title("3. Matriz de confusión", pad=30)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. CLI
# =============================================================================

def imprimir_hiperparametros(red: PerceptronMulticapa) -> None:
    print("\n--- Hiperparámetros ---")
    print(f"u (entradas)            : {red.u}")
    print(f"v (salidas)             : {red.v}")
    print(f"L (capas ocultas)       : {len(red.capas_ocultas)}")
    print(f"b (neuronas por capa)   : {red.capas_ocultas}")
    print(f"e (épocas acumuladas)   : {red.epocas}")
    print(f"tasa de aprendizaje     : {red.tasa_aprendizaje}")


def _pedir_entero(mensaje: str, minimo: int = 1) -> int:
    while True:
        try:
            valor = int(input(mensaje).strip())
            if valor >= minimo:
                return valor
            print(f"Debe ser un entero >= {minimo}.")
        except ValueError:
            print("Entrada inválida. Debe ingresar un número entero.")


def _pedir_ruta_existente(mensaje: str) -> Path:
    while True:
        ruta = Path(input(mensaje).strip().strip('"')).expanduser()
        if ruta.is_file():
            return ruta
        print(f"No se encontró el archivo: {ruta}")


def _cargar_dataset_validado(red: PerceptronMulticapa, mensaje: str):
    """Pide un archivo de datos, lo lee y valida; devuelve (X, y) o None si hay error."""
    ruta = _pedir_ruta_existente(mensaje)
    try:
        X, y = cargar_datos(ruta)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    error = validar_compatibilidad(red, X, y)
    if error:
        print(f"\n[ERROR] {error}")
        return None
    return X, y


def _crear_red() -> PerceptronMulticapa:
    print("\n--- Nueva red ---")
    u = _pedir_entero("Neuronas de entrada: ")
    v = _pedir_entero("Neuronas de salida (una por clase): ")
    L = _pedir_entero("Número de capas ocultas: ", minimo=0)
    b = _pedir_entero("Neuronas por capa oculta: ") if L > 0 else 0
    return PerceptronMulticapa(u, v, [b] * L)


def _accion_entrenar(red: PerceptronMulticapa) -> None:
    datos = _cargar_dataset_validado(red, "Archivo de entrenamiento: ")
    if datos is None:
        return
    X, y = datos
    epocas = _pedir_entero("Número de épocas: ")
    if red.clases is None:
        red.ajustar_clases(y)
    historial = red.entrenar(X, y, epocas)
    print(f"Entrenamiento terminado. MSE final: {historial[-1]:.6f}")
    imprimir_hiperparametros(red)
    mostrar_resultados(red, X, y, historial=historial, titulo="Resultado del entrenamiento")


def _accion_probar(red: PerceptronMulticapa) -> None:
    datos = _cargar_dataset_validado(red, "Archivo de prueba: ")
    if datos is None:
        return
    X, y = datos
    if red.clases is None:        # red cargada de CSV: el formato no guarda etiquetas
        red.ajustar_clases(y)
    precision = float(np.mean(red.predecir(X) == y))
    print(f"Exactitud en prueba: {precision * 100:.2f}%")
    imprimir_hiperparametros(red)
    mostrar_resultados(red, X, y, historial=None, titulo="Resultado de la prueba")


def main() -> None:
    print("========================================")
    print("      PERCEPTRÓN MULTICAPA (NumPy)      ")
    print("========================================")
    red: Optional[PerceptronMulticapa] = None
    while red is None:
        print("\n1. Crear nueva red\n2. Cargar red desde archivo CSV")
        opcion = input("Opción: ").strip()
        if opcion == "1":
            red = _crear_red()
        elif opcion == "2":
            try:
                red = cargar_red(_pedir_ruta_existente("Archivo de la red: "))
                print("Red cargada.")
            except Exception as e:
                print(f"No se pudo cargar la red: {e}")
        else:
            print("Opción inválida.")
    imprimir_hiperparametros(red)

    while True:
        print("\n1. Entrenar\n2. Probar\n3. Guardar red\n4. Salir")
        opcion = input("Opción: ").strip()
        if opcion == "1":
            _accion_entrenar(red)
        elif opcion == "2":
            _accion_probar(red)
        elif opcion == "3":
            ruta = Path(input("Ruta de guardado (ej. red_2d.csv): ").strip().strip('"')).expanduser()
            try:
                guardar_red(red, ruta)
                print(f"Red guardada en {ruta}")
            except OSError as e:
                print(f"No se pudo guardar: {e}")
        elif opcion == "4":
            print("Finalizando programa.")
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()