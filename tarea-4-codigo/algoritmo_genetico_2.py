# Nombre del integrante: Verónica Girón
# Cédula del integrante: 29908725

# Difference threshold: stop after a generation if the best fitness minus
# the worst fitness is less than or equal to the threshold.
# Mutation variability is the standard deviation of the random change.

import os
import runpy
import json
import random
import statistics

# Use the folder containing this program, regardless of the terminal folder.
PROGRAM_FOLDER = os.path.dirname(os.path.abspath(__file__))


def read_number(message, default, minimum, integer=False):
    while True:
        text = input(f"{message} [{default}]: ").strip()
        if text == "":
            text = str(default)
        try:
            if integer:
                value = int(text)
            else:
                value = float(text)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Introduce un número válido mayor o igual a {minimum}.")


def file_path(filename):
    return os.path.join(PROGRAM_FOLDER, filename)


def load_function(filename):
    # Read the selected Python file. Its definitions become a dictionary.
    path = file_path(filename)
    return runpy.run_path(path)


def evaluate_member(variables, function):
    # Unpack [x] or [x, y] into the arguments expected by fitness.
    result = float(function["fitness"](*variables))
    return {"variables": variables, "fitness": result}


def create_member(function):
    variables = []
    lower, upper = function["BOUNDS"]
    for _ in range(function["VARIABLE_COUNT"]):
        variables.append(random.uniform(lower, upper))
    return evaluate_member(variables, function)


def get_fitness(member):
    return member["fitness"]


def update_fitness(population, function):
    # The function file may have changed since the population was created.
    members = []
    for member in population["members"]:
        members.append(evaluate_member(member["variables"], function))
    members.sort(key=get_fitness, reverse=True)
    population["members"] = members


def create_population(filename, size=10, threshold=0.05,
                      generations=500, variability=1):
    parameters = {
        "population_size": size,
        "difference_threshold": threshold,
        "max_generations": generations,
        "mutation_variability": variability,
        "generations_elapsed": 0,
    }
    function = load_function(filename)
    members = [create_member(function) for _ in range(size)]
    members.sort(key=get_fitness, reverse=True)
    return {
        "function": filename,
        "parameters": parameters,
        "members": members,
    }


def select_parent(members):
    # Tournament: choose two members at random and keep the better one.
    first, second = random.sample(members, 2)
    return first if first["fitness"] >= second["fitness"] else second


def crossover(first_parent, second_parent):
    first_variables = first_parent["variables"]
    second_variables = second_parent["variables"]
    child_variables = []
    for i in range(len(first_variables)):
        weight = random.random()
        new_value = weight * first_variables[i] + (1 - weight) * second_variables[i]
        child_variables.append(new_value)
    return child_variables


def mutate(variables, variability, bounds):
    lower, upper = bounds
    for i in range(len(variables)):
        if random.random() < 0.5:
            variables[i] = variables[i] + random.gauss(0, variability)
        if variables[i] < lower:
            variables[i] = lower
        elif variables[i] > upper:
            variables[i] = upper
    return variables


def print_parameters(population):
    parameters = population["parameters"]
    print("\nFunción:", population["function"])
    print("Tamaño de población:", parameters["population_size"])
    print("Umbral de diferencia:", parameters["difference_threshold"])
    print("Máximo de generaciones:", parameters["max_generations"])
    print("Generaciones transcurridas:", parameters["generations_elapsed"])
    print("Variabilidad de mutación:", parameters["mutation_variability"])


def show_population(population):
    population["members"].sort(key=get_fitness, reverse=True)
    print_parameters(population)
    for position, member in enumerate(population["members"], 1):
        print(position, "Variables:", member["variables"], "Fitness:", member["fitness"])


def print_generation(population):
    scores = [member["fitness"] for member in population["members"]]
    maximum = max(scores)
    middle = statistics.median(scores)
    minimum = min(scores)
    generation = population["parameters"]["generations_elapsed"]
    print(f"Generation {generation} | Max: {maximum:.8f} | "
          f"Median: {middle:.8f} | Min: {minimum:.8f}")


def train_population(population):
    function = load_function(population["function"])
    update_fitness(population, function)
    parameters = population["parameters"]
    size = parameters["population_size"]

    while parameters["generations_elapsed"] < parameters["max_generations"]:
        members = population["members"]
        best = members[0]
        new_members = [{"variables": best["variables"][:], "fitness": best["fitness"]}]

        while len(new_members) < size:
            first_parent = select_parent(members)
            second_parent = select_parent(members)
            variables = crossover(first_parent, second_parent)
            variables = mutate(variables, parameters["mutation_variability"], function["BOUNDS"])
            new_members.append(evaluate_member(variables, function))

        new_members.sort(key=get_fitness, reverse=True)
        population["members"] = new_members
        parameters["generations_elapsed"] += 1

        print_generation(population)
        difference = new_members[0]["fitness"] - new_members[-1]["fitness"]
        if difference <= parameters["difference_threshold"]:
            print(f"Convergence reached: fitness difference = {difference:.8f}")
            break
    show_population(population)


def save_population(population, filename):
    path = file_path(filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(population, file, indent=2, allow_nan=False)
    print("Saved:", os.path.basename(path))


def load_population(filename):
    path = file_path(filename)
    with open(path, encoding="utf-8") as file:
        population = json.load(file)
    function = load_function(population["function"])
    update_fitness(population, function)
    show_population(population)
    return population


def create_population_menu():
    filename = input("Archivo de función [funcion_a_2.py]: ").strip() or "funcion_a_2.py"
    size = read_number("Tamaño de población", 10, 2, True)
    threshold = read_number("Umbral de diferencia", 0.05, 0)
    generations = read_number("Máximo de generaciones", 500, 1, True)
    variability = read_number("Variabilidad de mutación", 1, 0)
    population = create_population(filename, size, threshold, generations, variability)
    show_population(population)
    return population


def request_population():
    """Ask the user to create or load a population.

    Returns the population dict, or None if the user chooses to exit.
    """
    while True:
        print("\n=== ALGORITMO GENÉTICO ===")
        print("1. Crear población nueva")
        print("2. Cargar población desde archivo")
        print("0. Salir")
        option = input("Elija una opción: ").strip()
        try:
            if option == "0":
                return None
            if option == "1":
                return create_population_menu()
            if option == "2":
                filename = input("Nombre del archivo JSON (con .json): ").strip()
                return load_population(filename)
            print("Opción no válida, intente otra vez.")
        except (OSError, ValueError) as error:
            print("Error:", error)


def main():
    population = request_population()
    if population is None:
        return

    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Entrenar / continuar")
        print("2. Ver población y parámetros")
        print("3. Guardar población")
        print("0. Salir")
        option = input("Elija una opción: ").strip()
        try:
            if option == "0":
                return
            elif option == "1":
                parameters = population["parameters"]
                current_maximum = parameters["max_generations"]
                if parameters["generations_elapsed"] >= current_maximum:
                    parameters["max_generations"] = read_number(
                        "Nuevo máximo total de generaciones", current_maximum + 500,
                        current_maximum + 1, True)
                train_population(population)
            elif option == "2":
                show_population(population)
            elif option == "3":
                filename = input("Nombre del archivo JSON (con .json): ").strip()
                save_population(population, filename)
            else:
                print("Opción no válida, intente otra vez.")
        except (OSError, ValueError) as error:
            print("Error:", error)


if __name__ == "__main__":
    main()