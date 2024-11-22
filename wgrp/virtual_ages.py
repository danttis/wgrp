import numpy as np

from wgrp.r_functions import rep


def virtual_age(propagation, q, current_virtual_age=0, x=0) -> dict:
    """
    Virtual Age:
        Compute the virtual age of the system, according to the quality of the
        intervention system (summarized in the rejuvenation parameter `q`) and
        the previous history of the system itself (reflected in `current_virtual_age`).
        Further, the mixed virtual age function from Ferreira et al. (2015) is considered,
        leading to a linear combination of Kijima Type I and Type II virtual age models.

    Parameters:
        propagation : float
            Reflects the weight between Kijima Type I and Kijima Type II virtual age models
            intrinsic to the last intervention. `0 <= propagation <= 1` such that
            `propagation = 1` leads to Kijima Type I and `propagation = 0` leads to Kijima Type II.
        q : float
            The rejuvenation parameter. It allows one to study the quality of the intervention system.
            If `q = 0`, one has the Weibull-based Renewal Process - RP (in which each intervention
            usually brings the system to an 'as good as new' - AGAN condition). If `q = 1`,
            one has the Weibull-based Non-Homogeneous Poisson Process - NHPP (in which each
            intervention usually brings the system to an 'as bad as old' - ABAO condition). On the other hand,
            if `b > 1`, then `0 < q < 1` might reflect that each intervention usually brings the system
            to an intermediate condition, between AGAN and ABAO. Further, either `b < 1` and `q > 1`
            or `b > 1` and `q < 0` might reflect the situations in which each intervention usually
            brings the system to a 'better than in its beginning' condition.
        current_virtual_age : float
            The value of the virtual age underlying the system, reflecting its condition
            previously to `x`. If `current_virtual_age` is not determined, then it is assumed
            that the system is new, i.e. `current_virtual_age = 0`.
        x : float
            The time since the last intervention.

    Returns:
        A dictionary containing:
            'propagation': The input weight `propagation`
            'virtual_age': The virtual age of the system at time `x`

    Examples:
        >>> virtual_age(propagation=0.5, q=0.8, current_virtual_age=2, x=3)
        {'propagation': 0.5, 'virtualAge': 4.2}

    References:
        Ferreira RJ, Firmino PRA, Cristino CT (2015):
        A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes.
        PLoS ONE, 10(7), e0133772.
        https://doi.org/10.1371/journal.pone.0133772

    """
    KijimaI = current_virtual_age + q * x
    KijimaII = q * (current_virtual_age + x)
    virtualAge = propagation * KijimaI + (1 - propagation) * KijimaII
    return {'propagation': propagation, 'virtualAge': virtualAge}


def get_sample_virtual_ages(x, q, propagations):
    """
    Compute the virtual ages for a sequence of interventions.

    This function computes the virtual ages of a system for a series of interventions,
    taking into account the quality of the intervention system and the propagation
    weights between Kijima Type I and Type II virtual age models.

    Parameters:
        x : list of float
            The times since the last interventions.
        q : float
            The rejuvenation parameter. It allows one to study the quality of the intervention system.
            If `q = 0`, one has the Weibull-based Renewal Process - RP (in which each intervention
            usually brings the system to an 'as good as new' - AGAN condition). If `q = 1`,
            one has the Weibull-based Non-Homogeneous Poisson Process - NHPP (in which each
            intervention usually brings the system to an 'as bad as old' - ABAO condition). On the other hand,
            if `b > 1`, then `0 < q < 1` might reflect that each intervention usually brings the system
            to an intermediate condition, between AGAN and ABAO. Further, either `b < 1` and `q > 1`
            or `b > 1` and `q < 0` might reflect the situations in which each intervention usually
            brings the system to a 'better than in its beginning' condition.
        propagations : list of float or None
            The weights between Kijima Type I and Type II virtual age models intrinsic to each intervention.
            If `propagations` is `None`, it is assumed that all interventions have a propagation weight of -1.

    Returns:
        list of float
            The virtual ages of the system at each time in `x`.
    """
    n = len(x)
    virtualAges = [0] * n  # Inicializa uma lista de zeros com tamanho n
    previousVirtualAge = 0

    if propagations is None:
        propagations = rep(
            -1, n
        )   # Se propagations estiver vazio, preenche com -1

    for i in range(n):
        virtualAges_tmp = virtual_age(
            propagations[i], q, previousVirtualAge, x[i]
        )
        virtualAges[i] = virtualAges_tmp['virtualAge']
        previousVirtualAge = virtualAges[i]

    return virtualAges


def _get_virtual_ages_and_a(b, q, propagations, x) -> dict:
    """
    Compute virtual ages and 'a' parameter using given inputs.

    Args:
        b (float): Value of 'b' parameter.
        q (float): q value.
        propagations (list or None): List of propagations or None.
        x (list): List of input values.

    Returns:
        dict: Dictionary with 'a' parameter and list of virtual ages.
    """
    virtual_ages = get_sample_virtual_ages(x, q, propagations)
    n = len(x)
    sum_ = 0
    previous_virtual_age = 0
    a = 0
    if b != 0:
        for i in range(n):
            current_value = x[i] + previous_virtual_age
            if current_value < 0:
                # print(
                #     f'Warning: invalid value encountered at index {i}. current_value={current_value}, setting to 0.'
                # )
                current_value = 0
            if previous_virtual_age < 0:
                # print(
                #     f'Warning: invalid value encountered at index {i}. previous_virtual_age={previous_virtual_age}, setting to 0.'
                # )
                previous_virtual_age = 0

            current_value_b = np.power(current_value, b)
            previous_virtual_age_b = np.power(previous_virtual_age, b)
            sum_ += current_value_b - previous_virtual_age_b
            previous_virtual_age = virtual_ages[i]

        sum_ /= n
        a = np.power(sum_, 1 / b)
    return {'a': a, 'virtualAges': virtual_ages}
