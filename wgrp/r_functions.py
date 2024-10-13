import math
import random

import numpy as np


def c(*args) -> list:
    """
    Function that creates a list from a variable number of arguments.

    Args:
        Variable number of arguments to be transformed into a list.

    Returns:
        list: List containing all the received arguments.
    """
    return list(args)


def numeric(x: float):
    """
    Convert input to integer or list of integers.

    Args:
        x: Input value or list of values.

    Returns:
        int or list: Converted integer or list of integers.
    """
    if isinstance(x, list):
        return [int(i) for i in x]
    else:
        return int(x)


def rep(x: float, q: int) -> np.array:
    """
    Replicate value x q times.

    Args:
        x: Value to be replicated.
        q: Number of replications.

    Returns:
        numpy.ndarray: Array containing q instances of x.
    """
    return np.full(q, x)


def runif(x: int) -> list:
    """
    Generate a list of x random numbers uniformly distributed between 0 and 1.

    Args:
        x: Number of random numbers to generate.

    Returns:
        list: List of random numbers.
    """
    return [random.uniform(0, 1) for _ in range(x)]


def isfinite(x) -> bool:
    """
    Check if input is finite.

    Args:
        x: Input value or list of values.

    Returns:
        bool or list: True if input is finite, False otherwise.
    """
    if isinstance(x, list):
        return [math.isfinite(num) for num in x]
    else:
        return math.isfinite(x)
