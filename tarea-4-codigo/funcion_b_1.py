"""
Function B: z = 20 + x - 10*cos(2*pi*x) + y - 10*cos(2*pi*y)
"""
import math

BOUNDS = (-10, 10)
VARIABLE_COUNT = 2


def fitness(*variables):
    x, y = variables[0], variables[1]
    return 20.0 + x - 10.0 * math.cos(2.0 * math.pi * x) + y - 10.0 * math.cos(2.0 * math.pi * y)