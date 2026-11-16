# Nombre del integrante: Diego Goncalves
# Cédula del integrante: 30124687
"""
Interpretación del Umbral de Diferencia (difference_threshold):
Se utiliza como criterio de parada temprana por convergencia. Si la diferencia
entre el mejor fitness y el peor fitness de la población actual es menor o igual
al umbral establecido (max_fitness - min_fitness <= threshold), se asume que la
población convergió y el algoritmo finaliza su ejecución.
"""

import importlib.util
import json
import os
import random
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(filename):
    """
    Search the file both in the direct path and within the script's folder.
    Returns the absolute path if it exists, or the combined path with BASE_DIR.
    """
    clean_name = os.path.basename(filename)
    # 1. Try the route as the user put it
    if os.path.exists(filename):
        return os.path.abspath(filename)
    # 2. Try within the script directory (tarea-4-codigo/)
    script_dir_path = os.path.join(BASE_DIR, clean_name)
    if os.path.exists(script_dir_path):
        return script_dir_path
    # 3. Try subfolder tarea-4-codigo if executed from the root
    repo_subpath = os.path.join(os.getcwd(), "tarea-4-codigo", clean_name)
    if os.path.exists(repo_subpath):
        return repo_subpath

    # If it doesn't exist yet (e.g., for saving), return the path in BASE_DIR
    return script_dir_path


def load_function_module(filepath):
    """Dynamically loads a function definition file."""
    real_path = resolve_path(filepath)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath} (buscado en {real_path})")
    spec = importlib.util.spec_from_file_location("func_module", real_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_individual(variable_count, bounds):
    """Creates a random individual within bounds using real-value representation."""
    return [random.uniform(bounds[0], bounds[1]) for _ in range(variable_count)]


def evaluate(individual, fitness_func):
    """Calculates fitness for an individual."""
    return fitness_func(*individual)


def mutate(individual, variability, bounds):
    """Gaussian mutation bounded within allowed limits."""
    mutated = []
    for gene in individual:
        new_val = gene + random.gauss(0, variability)
        new_val = max(bounds[0], min(bounds[1], new_val))
        mutated.append(new_val)
    return mutated


def crossover(parent1, parent2):
    """Arithmetic blend crossover between two parents."""
    alpha = random.random()
    child = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent1, parent2)]
    return child


def tournament_select(population, fitnesses, k=3):
    """Selects the best individual among k random competitors."""
    selected_indices = random.sample(range(len(population)), k=min(k, len(population)))
    best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def sort_population(members):
    """Sorts members list descending by fitness."""
    return sorted(members, key=lambda m: m["fitness"], reverse=True)


def print_population_details(params, members):
    """Displays parameters and sorted population members."""
    print("PARÁMETROS DEL ALGORITMO:")
    print(f" - Tamaño de población:     {params['population_size']}")
    print(f" - Umbral de diferencia:    {params['difference_threshold']}")
    print(f" - Generaciones máximas:    {params['max_generations']}")
    print(f" - Generaciones ejecutadas: {params['generations_elapsed']}")
    print(f" - Variabilidad de mutación:{params['mutation_variability']}")
    print(f"{'Posición':<10}{'Variables':<35}{'Fitness'}")
    print("-" * 65)
    for idx, member in enumerate(members, 1):
        vars_str = ", ".join(f"{v:8.4f}" for v in member["variables"])
        print(f"{idx:<10}[{vars_str:<32}] {member['fitness']:10.4f}")
    print("=" * 65 + "\n")


def run_genetic_algorithm(func_module, func_filename, params, initial_members=None):
    """Executes the genetic algorithm loop."""
    bounds = func_module.BOUNDS
    var_count = func_module.VARIABLE_COUNT
    fitness_func = func_module.fitness

    pop_size = params["population_size"]
    threshold = params["difference_threshold"]
    max_gens = params["max_generations"]
    variability = params["mutation_variability"]
    elapsed_so_far = params.get("generations_elapsed", 0)

    # Initiate population either from provided members or randomly
    if initial_members:
        population = [list(m["genes"]) for m in initial_members]
    else:
        population = [create_individual(var_count, bounds) for _ in range(pop_size)]

    gens_run = 0

    print("\n--- INICIO DE ENTRENAMIENTO ---")
    for gen in range(1, max_gens + 1):
        fitnesses = [evaluate(ind, fitness_func) for ind in population]
        gens_run = gen

        max_fit = max(fitnesses)
        min_fit = min(fitnesses)
        med_fit = statistics.median(fitnesses)

        total_gen = elapsed_so_far + gen
        print(f"Gen {total_gen:4d} | Máx: {max_fit:10.4f} | Mediana: {med_fit:10.4f} | Mín: {min_fit:10.4f}")

        # Check for convergence based on the difference threshold
        if (max_fit - min_fit) <= threshold:
            print(f">> Convergencia alcanzada en la generación {total_gen} (Diferencia: {max_fit - min_fit:.5f} <= {threshold})")
            break

        # Create new population using elitism, selection, crossover, and mutation
        best_individual = population[fitnesses.index(max_fit)]
        new_population = [list(best_individual)]

        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, variability, bounds)
            new_population.append(child)

        population = new_population

    final_fitnesses = [evaluate(ind, fitness_func) for ind in population]
    members = [
        {"variables": [round(v, 4) for v in ind], "fitness": round(fit, 4), "genes": [round(v, 4) for v in ind]}
        for ind, fit in zip(population, final_fitnesses)
    ]
    members = sort_population(members)

    params["generations_elapsed"] = elapsed_so_far + gens_run

    print_population_details(params, members)
    return {
        "function": os.path.basename(func_filename),
        "parameters": params,
        "members": members,
    }


def save_to_file(data, filepath):
    """Saves population structure to JSON file."""
    real_path = resolve_path(filepath)
    with open(real_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Población guardada exitosamente en: {real_path}")


def load_from_file(filepath):
    """Loads population structure from JSON file."""
    real_path = resolve_path(filepath)
    with open(real_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    while True:
        print("\nALGORITMO GENÉTICO")
        print("1. Crear y entrenar nueva población")
        print("2. Cargar población desde archivo (mostrar / continuar)")
        print("3. Salir")
        choice = input("Seleccione una opción: ").strip()

        if choice == "1":
            func_file = input("Archivo de función [funcion_a_1.py]: ").strip()
            if not func_file:
                func_file = "funcion_a_1.py"

            real_func_path = resolve_path(func_file)
            if not os.path.exists(real_func_path):
                print(f"Error: No se encontró '{func_file}' en ninguna ruta conocida.")
                continue

            module = load_function_module(real_func_path)

            pop_size_input = input("Tamaño de población [10]: ").strip()
            pop_size = int(pop_size_input) if pop_size_input else 10

            thresh_input = input("Umbral de diferencia [0.05]: ").strip()
            threshold = float(thresh_input) if thresh_input else 0.05

            gens_input = input("Máximo de generaciones [500]: ").strip()
            max_gens = int(gens_input) if gens_input else 500

            var_input = input("Variabilidad de mutación [1.0]: ").strip()
            variability = float(var_input) if var_input else 1.0

            params = {
                "population_size": pop_size,
                "difference_threshold": threshold,
                "max_generations": max_gens,
                "mutation_variability": variability,
                "generations_elapsed": 0,
            }

            data = run_genetic_algorithm(module, func_file, params)

            save_opt = input("¿Desea guardar la población en un archivo JSON? (s/n): ").strip().lower()
            if save_opt == "s":
                out_path = input("Nombre de archivo destino (ej. population_1_a_low.json): ").strip()
                save_to_file(data, out_path)

        elif choice == "2":
            in_path = input("Nombre de archivo a cargar: ").strip()
            real_in_path = resolve_path(in_path)
            if not os.path.exists(real_in_path):
                print(f"Error: No se encontró el archivo '{in_path}'.")
                continue

            data = load_from_file(real_in_path)
            print("\n>> Población cargada:")
            print_population_details(data["parameters"], data["members"])

            cont_opt = input("¿Desea continuar entrenando esta población? (s/n): ").strip().lower()
            if cont_opt == "s":
                func_file = data.get("function")
                real_func_path = resolve_path(func_file)
                if not os.path.exists(real_func_path):
                    func_file = input(f"No se encontró '{func_file}'. Ingrese el nombre del archivo de función: ").strip()
                    real_func_path = resolve_path(func_file)

                module = load_function_module(real_func_path)

                gens_input = input("Generaciones adicionales a entrenar [100]: ").strip()
                add_gens = int(gens_input) if gens_input else 100

                var_input = input(f"Variabilidad de mutación [{data['parameters']['mutation_variability']}]: ").strip()
                if var_input:
                    data["parameters"]["mutation_variability"] = float(var_input)

                data["parameters"]["max_generations"] = add_gens
                new_data = run_genetic_algorithm(module, func_file, data["parameters"], initial_members=data["members"])

                save_opt = input("¿Desea guardar los cambios en un archivo JSON? (s/n): ").strip().lower()
                if save_opt == "s":
                    out_path = input(f"Nombre de archivo destino [{in_path}]: ").strip()
                    save_to_file(new_data, out_path if out_path else in_path)

        elif choice == "3":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida.")

main()
