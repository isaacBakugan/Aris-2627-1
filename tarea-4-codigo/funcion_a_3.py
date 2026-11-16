"""Function a: y = sin^2(x) / x, searched on x in [-10, 10].

At x = 0 the expression is 0/0; its limit is 0, so that value is returned instead
of raising ZeroDivisionError.
"""

import math

BOUNDS = (-10, 10)  # limits of every variable
VARIABLE_COUNT = 1  # one variable: x


def fitness(*variables: float) -> float:
    """Return y = sin^2(x) / x (higher is better)."""
    (x,) = variables
    if x == 0:
        return 0.0  # lim_{x -> 0} sin^2(x) / x = 0
    return math.sin(x) ** 2 / x
