import math

BOUNDS = (-10, 10)
VARIABLE_COUNT = 2


def fitness(x, y):
    # Calculate the contribution of each variable separately.
    x_part = x - 10 * math.cos(2 * math.pi * x)
    y_part = y - 10 * math.cos(2 * math.pi * y)
    result = 20 + x_part + y_part
    return result
