"""
Function A: y = sin^2(x) / x
"""
import math

BOUNDS = (-10, 10)
VARIABLE_COUNT = 1


def fitness(*variables):
    x = variables[0]
    if abs(x) < 1e-9:
        return 0.0  # Limit of sin^2(x)/x when x -> 0 is 0
    return (math.sin(x) ** 2) / x