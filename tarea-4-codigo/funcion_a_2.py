import math

BOUNDS = (-10, 10)
VARIABLE_COUNT = 1


def fitness(x):
    # Use the limit at zero to avoid dividing by zero.
    if x == 0:
        return 0.0

    sine = math.sin(x)
    numerator = sine ** 2
    result = numerator / x
    return result
