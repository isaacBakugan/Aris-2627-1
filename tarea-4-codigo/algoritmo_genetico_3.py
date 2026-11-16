# Nombre del integrante: Máximo Mujica
# Cédula del integrante: 30727257

"""Algoritmo genético con genes reales (un gen por variable) que maximiza la función de un archivo aparte.
Umbral de diferencia: criterio de parada; si máximo - mínimo queda bajo el umbral durante
CONVERGENCE_PATIENCE generaciones seguidas, el entrenamiento termina antes del máximo de generaciones."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import numpy as np

DEFAULT_POPULATION_SIZE = 10
DEFAULT_DIFFERENCE_THRESHOLD = 0.05
DEFAULT_MAX_GENERATIONS = 500
DEFAULT_MUTATION_VARIABILITY = 1.0
DEFAULT_FUNCTION_FILE = "funcion_a_3.py"

# Parámetros fijos del operador genético
ELITE_COUNT = 1  # miembros que pasan sin cambios
TOURNAMENT_SIZE = 3
BLX_ALPHA = 0.5
MUTATION_RATE = 0.5  # probabilidad de mutar cada gen
CONVERGENCE_PATIENCE = 25  # generaciones seguidas bajo el umbral para parar

PROGRAM_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ObjectiveFunction:
    # Función objetivo cargada desde su archivo
    file_name: str
    bounds: tuple[float, float]
    variable_count: int
    fitness: Callable[..., float]

    def evaluate(self, genes: np.ndarray) -> np.ndarray:
        # Evalúa cada miembro; un valor inválido recibe -inf para que nunca sea el mejor
        values = np.empty(genes.shape[0], dtype=float)
        for i, row in enumerate(genes):
            try:
                value = float(self.fitness(*(float(v) for v in row)))
            except (ZeroDivisionError, ValueError, OverflowError):
                value = -math.inf
            values[i] = value if not math.isnan(value) else -math.inf
        return values


def resolve_path(name: str, extra_dirs: tuple[Path, ...] = ()) -> Path:
    # Busca el archivo en las carpetas dadas, la actual y la del programa (sin rutas absolutas fijas)
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate
    for directory in (*extra_dirs, Path.cwd(), PROGRAM_DIR):
        path = directory / candidate
        if path.is_file():
            return path
    return PROGRAM_DIR / candidate


def load_objective_function(name: str, extra_dirs: tuple[Path, ...] = ()) -> ObjectiveFunction:
    # Importa el archivo de función y valida el contrato BOUNDS / VARIABLE_COUNT / fitness
    path = resolve_path(name, extra_dirs)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de función '{name}'.")

    spec = importlib.util.spec_from_file_location(f"objective_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar '{path.name}'.")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attribute in ("BOUNDS", "VARIABLE_COUNT", "fitness"):
        if not hasattr(module, attribute):
            raise AttributeError(f"'{path.name}' no define {attribute}.")
    low, high = (float(b) for b in module.BOUNDS)
    variable_count = int(module.VARIABLE_COUNT)
    if not low < high:
        raise ValueError(f"BOUNDS inválidos en '{path.name}': {module.BOUNDS}.")
    if variable_count < 1:
        raise ValueError(f"VARIABLE_COUNT inválido en '{path.name}': {variable_count}.")
    if not callable(module.fitness):
        raise TypeError(f"'fitness' en '{path.name}' no es una función.")

    return ObjectiveFunction(path.name, (low, high), variable_count, module.fitness)


@dataclass
class GAParameters:
    population_size: int = DEFAULT_POPULATION_SIZE
    difference_threshold: float = DEFAULT_DIFFERENCE_THRESHOLD
    max_generations: int = DEFAULT_MAX_GENERATIONS
    mutation_variability: float = DEFAULT_MUTATION_VARIABILITY
    generations_elapsed: int = 0

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("El tamaño de la población debe ser al menos 2.")
        if self.difference_threshold < 0:
            raise ValueError("El umbral de diferencia no puede ser negativo.")
        if self.max_generations < 1:
            raise ValueError("El número máximo de generaciones debe ser al menos 1.")
        if self.mutation_variability <= 0:
            raise ValueError("La variabilidad de la mutación debe ser mayor que 0.")


GenerationCallback = Callable[[int, float, float, float], None]


class GeneticAlgorithm:
    # Población (arreglo de forma tamaño x variables) y operadores evolutivos
    def __init__(
        self,
        function: ObjectiveFunction,
        parameters: GAParameters,
        genes: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        parameters.validate()
        self.function = function
        self.parameters = parameters
        self.rng = np.random.default_rng(seed)
        low, high = function.bounds
        shape = (parameters.population_size, function.variable_count)
        if genes is None:
            genes = self.rng.uniform(low, high, size=shape)
        genes = np.asarray(genes, dtype=float)
        if genes.shape != shape:
            raise ValueError(f"Los genes tienen forma {genes.shape}, se esperaba {shape}.")
        self.genes = np.clip(genes, low, high)
        self.fitness = function.evaluate(self.genes)
        self._sort()

    def _sort(self) -> None:
        # Mantiene la población ordenada de mejor a peor
        order = np.argsort(-self.fitness, kind="stable")
        self.genes = self.genes[order]
        self.fitness = self.fitness[order]

    def statistics(self) -> tuple[float, float, float]:
        # (máximo, mediana, mínimo) del valor de la función
        return float(self.fitness.max()), float(np.median(self.fitness)), float(self.fitness.min())

    def has_converged(self) -> bool:
        maximum, _, minimum = self.statistics()
        return (maximum - minimum) < self.parameters.difference_threshold

    def _tournament(self, count: int) -> np.ndarray:
        # Selección por torneo: gana el mejor de TOURNAMENT_SIZE miembros al azar
        size = self.parameters.population_size
        contestants = self.rng.integers(0, size, size=(count, min(TOURNAMENT_SIZE, size)))
        winners = np.argmax(self.fitness[contestants], axis=1)
        return contestants[np.arange(count), winners]

    def _crossover(self, parents_a: np.ndarray, parents_b: np.ndarray) -> np.ndarray:
        # Cruce BLX-alfa: cada gen hijo cae en el intervalo de los padres, ampliado
        lower = np.minimum(parents_a, parents_b)
        upper = np.maximum(parents_a, parents_b)
        spread = upper - lower
        return self.rng.uniform(lower - BLX_ALPHA * spread, upper + BLX_ALPHA * spread)

    def _mutate(self, children: np.ndarray) -> np.ndarray:
        # Mutación gaussiana; la desviación estándar es la variabilidad de la mutación
        noise = self.rng.normal(0.0, self.parameters.mutation_variability, size=children.shape)
        mask = self.rng.random(children.shape) < MUTATION_RATE
        return children + noise * mask

    def step(self) -> None:
        # Una generación: elitismo, selección, cruce, mutación y reordenamiento
        size = self.parameters.population_size
        elite_count = min(ELITE_COUNT, size)
        child_count = size - elite_count

        parents_a = self.genes[self._tournament(child_count)]
        parents_b = self.genes[self._tournament(child_count)]
        children = self._mutate(self._crossover(parents_a, parents_b))
        children = np.clip(children, *self.function.bounds)

        self.genes = np.vstack([self.genes[:elite_count], children])
        self.fitness = np.concatenate(
            [self.fitness[:elite_count], self.function.evaluate(children)]
        )
        self._sort()
        self.parameters.generations_elapsed += 1

    def run(self, generations: int, on_generation: Optional[GenerationCallback] = None) -> bool:
        # Devuelve True si se detuvo por convergencia antes de terminar las generaciones
        converged_streak = 0
        for _ in range(generations):
            self.step()
            if on_generation is not None:
                on_generation(self.parameters.generations_elapsed, *self.statistics())
            converged_streak = converged_streak + 1 if self.has_converged() else 0
            if converged_streak >= CONVERGENCE_PATIENCE:
                return True
        return False


def _json_number(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def save_population(path: Path, algorithm: GeneticAlgorithm) -> None:
    # Guarda el JSON del contrato: valores decodificados, miembros de mejor a peor
    members = [
        {
            "variables": [float(v) for v in genes],
            "fitness": _json_number(fit),
            "genes": {"encoding": "real", "values": [float(v) for v in genes]},
        }
        for genes, fit in zip(algorithm.genes, algorithm.fitness)
    ]
    document = {
        "function": algorithm.function.file_name,
        "parameters": asdict(algorithm.parameters),
        "members": members,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class LoadedPopulation:
    function_file: str
    parameters: GAParameters
    variables: np.ndarray
    fitness: np.ndarray
    algorithm: Optional[GeneticAlgorithm]  # None si no se encuentra el archivo de función
    warning: Optional[str] = None


def load_population(path: Path) -> LoadedPopulation:
    # Lee el JSON y reconstruye el algoritmo para poder seguir entrenando
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document["parameters"]
    parameters = GAParameters(
        population_size=int(raw["population_size"]),
        difference_threshold=float(raw["difference_threshold"]),
        max_generations=int(raw["max_generations"]),
        mutation_variability=float(raw["mutation_variability"]),
        generations_elapsed=int(raw.get("generations_elapsed", 0)),
    )
    members = document["members"]
    if len(members) != parameters.population_size:
        raise ValueError(
            f"El archivo trae {len(members)} miembros pero population_size = "
            f"{parameters.population_size}."
        )

    def member_genes(member: dict) -> list[float]:
        genes = member.get("genes")
        if isinstance(genes, dict) and "values" in genes:
            return [float(v) for v in genes["values"]]
        return [float(v) for v in member["variables"]]

    variables = np.array([[float(v) for v in m["variables"]] for m in members], dtype=float)
    fitness = np.array(
        [-math.inf if m.get("fitness") is None else float(m["fitness"]) for m in members]
    )
    genes = np.array([member_genes(m) for m in members], dtype=float)

    function_file = str(document["function"])
    try:
        function = load_objective_function(function_file, extra_dirs=(path.resolve().parent,))
        algorithm = GeneticAlgorithm(function, parameters, genes=genes)
        return LoadedPopulation(function_file, parameters, algorithm.genes, algorithm.fitness, algorithm)
    except (OSError, ImportError, AttributeError, ValueError, TypeError) as error:
        return LoadedPopulation(function_file, parameters, variables, fitness, None, str(error))


def print_generation(generation: int, maximum: float, median: float, minimum: float) -> None:
    print(
        f"Generación {generation:5d} | máximo: {maximum:12.6f} | "
        f"mediana: {median:12.6f} | mínimo: {minimum:12.6f}"
    )


def print_summary(function_file: str, parameters: GAParameters,
                  variables: np.ndarray, fitness: np.ndarray) -> None:
    print("\n================ PARÁMETROS ================")
    print(f"Función:                      {function_file}")
    print(f"Tamaño de la población:       {parameters.population_size}")
    print(f"Umbral de diferencia:         {parameters.difference_threshold}")
    print(f"Máximo de generaciones:       {parameters.max_generations}")
    print(f"Generaciones transcurridas:   {parameters.generations_elapsed}")
    print(f"Variabilidad de la mutación:  {parameters.mutation_variability}")
    print("\n====== MIEMBROS (de mejor a peor) ======")
    names = ["x", "y", "z", "w"]
    for rank, (row, fit) in enumerate(zip(variables, fitness), start=1):
        labels = names if len(row) <= len(names) else [f"v{i + 1}" for i in range(len(row))]
        values = ", ".join(f"{labels[i]} = {v:10.6f}" for i, v in enumerate(row))
        print(f"{rank:3d}. {values}  ->  función = {fit:.6f}")
    print()


def show_algorithm(algorithm: GeneticAlgorithm) -> None:
    print_summary(algorithm.function.file_name, algorithm.parameters, algorithm.genes, algorithm.fitness)


def ask(prompt: str, cast: Callable[[str], object], default: object) -> object:
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if not text:
            return default
        try:
            return cast(text)
        except ValueError:
            print("Valor inválido, intente de nuevo.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "S/n" if default else "s/N"
    text = input(f"{prompt} ({suffix}): ").strip().lower()
    return default if not text else text.startswith("s")


def train(algorithm: GeneticAlgorithm) -> None:
    generations = int(ask("Generaciones a ejecutar", int, algorithm.parameters.max_generations))
    if generations < 1:
        print("Debe ejecutar al menos una generación.")
        return
    print()
    converged = algorithm.run(generations, on_generation=print_generation)
    if converged:
        print(
            f"\nLa población convergió (máximo - mínimo < "
            f"{algorithm.parameters.difference_threshold} durante {CONVERGENCE_PATIENCE} "
            f"generaciones seguidas); entrenamiento detenido."
        )
    show_algorithm(algorithm)


def create_population() -> Optional[GeneticAlgorithm]:
    print("\n--- Nueva población (Enter = valor por defecto) ---")
    function_file = str(ask("Archivo de la función", str, DEFAULT_FUNCTION_FILE))
    try:
        function = load_objective_function(function_file)
    except (OSError, ImportError, AttributeError, ValueError, TypeError) as error:
        print(f"Error: {error}")
        return None
    print(f"Función cargada: {function.variable_count} variable(s) en {function.bounds}.")

    parameters = GAParameters(
        population_size=int(ask("Tamaño de la población", int, DEFAULT_POPULATION_SIZE)),
        difference_threshold=float(ask("Umbral de diferencia", float, DEFAULT_DIFFERENCE_THRESHOLD)),
        max_generations=int(ask("Número máximo de generaciones", int, DEFAULT_MAX_GENERATIONS)),
        mutation_variability=float(ask("Variabilidad de la mutación", float, DEFAULT_MUTATION_VARIABILITY)),
    )
    seed_text = input("Semilla aleatoria (Enter = aleatoria): ").strip()
    try:
        seed = int(seed_text) if seed_text else None
        algorithm = GeneticAlgorithm(function, parameters, seed=seed)
    except ValueError as error:
        print(f"Error: {error}")
        return None
    print("Población creada.")
    if ask_yes_no("¿Desea entrenarla ahora?"):
        train(algorithm)
    return algorithm


def load_population_menu() -> Optional[GeneticAlgorithm]:
    name = input("Ruta del archivo de población (.json): ").strip()
    path = resolve_path(name)
    try:
        loaded = load_population(path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
        print(f"Error al cargar la población: {error}")
        return None
    print(f"Población cargada desde '{path.name}'.")
    print_summary(loaded.function_file, loaded.parameters, loaded.variables, loaded.fitness)
    if loaded.algorithm is None:
        print(f"Aviso: no se puede continuar el entrenamiento ({loaded.warning}).")
    return loaded.algorithm


def save_population_menu(algorithm: GeneticAlgorithm) -> None:
    name = input("Nombre del archivo a guardar (.json): ").strip()
    if not name:
        print("Nombre vacío, no se guardó nada.")
        return
    path = Path(name)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    try:
        save_population(path, algorithm)
    except OSError as error:
        print(f"Error al guardar: {error}")
        return
    print(f"Población guardada en '{path}'.")


def main() -> None:
    print("==============================================")
    print("        ALGORITMO GENÉTICO - TAREA 4          ")
    print("==============================================")
    algorithm: Optional[GeneticAlgorithm] = None
    while True:
        print("\n1. Crear población nueva")
        print("2. Cargar población desde archivo")
        print("3. Entrenar población actual")
        print("4. Guardar población actual")
        print("5. Mostrar población actual")
        print("6. Salir")
        try:
            option = input("Opción: ").strip()
        except EOFError:
            print()
            break
        if option == "1":
            algorithm = create_population() or algorithm
        elif option == "2":
            algorithm = load_population_menu() or algorithm
        elif option in ("3", "4", "5") and algorithm is None:
            print("Primero cree o cargue una población.")
        elif option == "3":
            train(algorithm)
        elif option == "4":
            save_population_menu(algorithm)
        elif option == "5":
            show_algorithm(algorithm)
        elif option == "6":
            print("Finalizando programa.")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido.")
        sys.exit(0)
