import numpy as np

from mwgrp.base_functions import Parameters
from mwgrp.r_functions import *
from mwgrp.virtual_ages import get_sample_virtual_ages, virtual_age


def ic_wgrp(optimum_obj: dict, x) -> dict:
    """
    Compute classical Information Criteria values intrinsic to the Weibull-based
    Renewal Process (WGRP) model under study.

    The Akaike information criterion (AIC), corrected AIC (AICc), and Bayesian
    information criterion (BIC) are computed.

    Parameters:
        optimum_obj (dict):
            A WGRP model. An object returned from the `get_mle_objs` function call.
        x (list of float):
            The time between events dataset.

    Returns:
        dict:
            A dictionary containing:
            - 'AIC': Akaike Information Criterion (AIC) value.
            - 'AICc': Corrected Akaike Information Criterion (AICc) value.
            - 'BIC': Bayesian Information Criterion (BIC) value.
            - 'logLik': Maximum log-likelihood of the WGRP model fitted to the data.

    Examples:
        >>> parameters = Parameters()
        >>> optimumObj = {
        ...     'parameters': {'formalism': parameters.FORMALISM['RP']},
        ...     'optimum': [10],
        ...     'optimum_value' : 10
        ... }
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> result = ic_wgrp(optimumObj, x)
        >>> result['AIC'] == -14.0
        False
        >>> np.isfinite(result['AICc'])
        np.True_
        >>> print(result['BIC'])
        -16.7811241751318

    References:
        - Akaike H (1974):
        A New Look at the Statistical Model Identification.
        IEEE Transactions on Automatic Control, 19(6), 716-723. https://doi.org/10.1109/TAC.1974.1100705
        - Stone M (1977):
        An Asymptotic Equivalence of Choice of Model by Cross-Validation and Akaike's
        Criterion.
        Journal of the Royal Statistical Society. Series B (Methodological), 39(1), 44-47. https://doi.org/10.1111/j.2517-6161.1977.tb01606.x
        - Schwarz G (1978):
        Estimating the Dimension of a Model.
        The Annals of Statistics, 6(2), 461-464. https://doi.org/10.1214/aos/1176344136
    """
    parameters = Parameters()
    nPar = -1
    n = len(x)
    formalism = optimum_obj['parameters']['formalism']
    if formalism == parameters.FORMALISM['RP']:
        nPar = 2
    elif formalism == parameters.FORMALISM['NHPP']:
        nPar = 2
    elif formalism == parameters.FORMALISM['KIJIMA_I']:
        nPar = 3
    elif formalism == parameters.FORMALISM['KIJIMA_II']:
        nPar = 3
    elif formalism == parameters.FORMALISM['INTERVENTION_TYPE']:
        nPar = 5
    elif formalism == parameters.FORMALISM['GENERIC_PROPAGATION']:
        nPar = 3 + n
    AIC = -2 * optimum_obj['optimum_value']
    AIC = AIC + 2 * nPar

    try:   # nesessario revisar
        AICc = AIC + 2 * nPar * (nPar + 1) / (n - nPar - 1)
    except ZeroDivisionError:
        AICc = -np.inf   # necessario revisar

    BIC = -2 * optimum_obj['optimum_value'] + nPar * np.log(n)
    return {'AIC': AIC, 'AICc': AICc, 'BIC': BIC}


def qwgrp(
    n: int,
    a: float,
    b: float,
    q: float,
    propagations: list,
    reliabilities: list = None,
    previous_virtual_age: float = 0,
) -> dict:
    """
    Inverse Generation of WGRP Samples

    Quantile function for the WGRP (Weibull-based Generalized Renewal Process)
    with scale parameter `a`, shape parameter `b`, rejuvenation parameter `q`,
    vector of mixed Kijima (MK) coefficients `propagations`, vector of desired
    probabilities `reliabilities`, and starting virtual age `previous_virtual_age`.
    MK coefficients allow one to distinguish the impact of preventive and corrective
    interventions.

    Parameters:
        n (int):
            The number of interventions taken into account. Due to the usual number of parameters
            of the model (`a`, `b`, `q`, `MK_{PM}`, and `MK_{CM}`), one must work with `n > 6`.
        a (float):
            The scale parameter. Similarly to the Weibull distribution, one must work with `a > 0`.
        b (float):
            The shape parameter. Similarly to the Weibull distribution, one must work with `b > 0`.
            This parameter reflects the stage faced by the system under study, whether stable, improving
            or deteriorating. If `b = 1`, one has an exponential distribution underlying the process.
            It implies in a stable system, i.e. without memory (in which the intervention rate is constant
            through the time). In turn, if `b < 1`, one usually has an improving system (in which the
            intervention rate decreases as time evolves). Finally, if `b > 1`, one usually has a
            deteriorating system (in which the intervention rate increases as time evolves). However,
            the meaning of `b` might change depending on the value of `q`.
        q (float):
            The rejuvenation parameter. It allows one to study the quality of the intervention system.
            If `q = 0`, one has the Weibull-based Renewal Process - RP (in which each intervention usually
            brings the system to an 'as good as new' - AGAN condition). If `q = 1`, one has the
            Weibull-based Non-Homogeneous Poisson Process - NHPP (in which each intervention usually
            brings the system to an 'as bad as old' - ABAO condition). On the other hand, if `b > 1`, then
            `0 < q < 1` might reflect that each intervention usually brings the system to an intermediate
            condition, between AGAN and ABAO. Further, either `b < 1` and `q > 1` or `b > 1` and `q < 0`
            might reflect the situations in which each intervention usually brings the system to a 'better
            than in its beginning' condition.
        propagations (list of float):
            A numeric vector of size `n`. Let `y_i` be the type of the i-th intervention (e.g. a preventive
            maintenance - PM or a corrective maintenance - CM). Further, let `c(y_i)` be the i-th element
            of `propagations`, related to the intervention type `y_i`. Thus, `c(y_i)` reflects the weight
            between Kijima Type I and Kijima Type II virtual age models intrinsic to `y_i`. Then,
            `0 <= c(y_i) <= 1` in such a way that `c(y_i) = 1` leads to Kijima Type I and `c(y_i) = 0`
            leads to Kijima Type II.
        reliabilities  (list of float) :
            A numeric vector of size `n`. The probabilities under which the WGRP quantiles must be computed.
            If `reliabilities` is not set, then a random sequence is internally created.
        previous_virtual_age (float):
            The starting value of the virtual age underlying the system, reflecting its initial condition,
            according to the virtual age concept. If `previous_virtual_age` is not set, then it is assumed
            that the system is new, i.e., `previous_virtual_age = 0`.

    Returns:
        A dictionary containing:
        - 'reliabilities': The probabilities under which the WGRP quantiles have been computed.
        - 'times': The times between interventions related to `reliabilities`.
        - 'virtualAges': The virtual ages related to `times`.

    Examples:
        >>> n = 10
        >>> a = 10
        >>> b = 2
        >>> q = 0.5
        >>> event_types = np.random.choice(["CM", "PM"], size=n, replace=True)
        >>> propagations = np.where(event_types == "CM", 0.8, 0.3)
        >>> reliabilities = np.full(n, 0.5)
        >>> previousVirtualAge = 10
        >>> result = qwgrp(n, a, b, q, propagations, reliabilities, previousVirtualAge)
        >>> len(result['reliabilities']) == n
        True
        >>> len(result['times']) == n
        True
        >>> len(result['virtualAges']) == n
        True

    References:
        - Ferreira RJ, Firmino PRA, Cristino CT (2015):
        A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes.
        PLoS ONE, 10(7), e0133772. https://doi.org/10.1371/journal.pone.0133772
        - Felix J, Firmino PRA, Ferreira RJ (2016):
        Kernel Density Estimation and Applications.
        Statistics, 50(3), 123-145. https://doi.org/10.1007/s00362-015-0704-5
        - Felix J, Firmino PRA, Ferreira RJ (2019):
        A Tool for Reliability Data Analysis.
        Applied Stochastic Models in Business and Industry, 35(4), 761-776. https://doi.org/10.1002/asmb.2396
        - Firmino PRA, Ferreira RJ (2021):
        Generalized Models in Reliability.
        Reliability Engineering & System Safety, 207, 107325. https://doi.org/10.1016/j.ress.2021.107325

    """

    x = np.zeros(n)
    virtualAges = np.zeros(n)

    if reliabilities is None:
        reliabilities = runif(n)
    elif not isinstance(reliabilities, np.ndarray):
        reliabilities = np.array(reliabilities)

    for i in range(n):
        u = reliabilities[i % len(reliabilities)]

        if np.isscalar(u) and u == 0:
            u = runif(1)[0]

        aux = (
            a * ((previous_virtual_age / a) ** b - np.log(u)) ** (1 / b)
            - previous_virtual_age
        )

        if np.isfinite(aux) and aux > 0:
            x[i] = aux

        virtualAge_result = virtual_age(
            propagations[i], q, previous_virtual_age, x[i]
        )
        virtualAges_tmp = virtualAge_result['virtualAge']
        virtualAges[i] = virtualAges_tmp
        previous_virtual_age = virtualAges[i]

    return {
        'reliabilities': reliabilities,
        'times': x,
        'virtualAges': virtualAges,
    }


def dwgrp(x, a: float, b: float, v: float, log: bool = True):
    """
    WGRP Cumulative Distribution Function
    WGRP Density
    The PDF (Probability Density Function) of the WGRP (Weibull-based Generalized Renewal Process) distribution,
    with scale parameter `a`, shape parameter `b`, and virtual age `v`, at a given time set `x`.

    Parameters:
        x (list of float):
            The times at which the WGRP PDF must be computed. Values must be greater than 0.
        a (float):
            The scale parameter. Similarly to the Weibull distribution, one must work with `a > 0`.
        b (float):
            The shape parameter. Similarly to the Weibull distribution, one must work with `b > 0`.
            This parameter reflects the stage faced by the system under study, whether stable, improving or deteriorating.
            If `b = 1`, one has an exponential distribution underlying the process. It implies in a stable system, i.e.
            without memory (in which the intervention rate is constant through the time). In turn, if `b < 1`, one usually
            has an improving system (in which the intervention rate decreases as time evolves). Finally, if `b > 1`, one
            usually has a deteriorating system (in which the intervention rate increases as time evolves). However, the
            meaning of `b` might change depending on the value of the rejuvenation parameter (`q`).
        v (float):
            The virtual age of the system prior to `x`. For general purposes, `v` is computed mainly considering the
            rejuvenation parameter (`q`).
        log (bool):
            If True (default), the PDFs are given at log-scale.

    Returns :
        list of float The values of the WGRP PDF at `x`.

    Examples:
        >>> dwgrp(2, 1, 2, 1)
        np.float64(-6.2082405307719455)
        >>> dwgrp(2, 0, 2, -1)  # This should handle invalid v gracefully
        -inf
        >>> dwgrp(2, -1, 2, 1)  # This should handle invalid a gracefully
        -inf

    References:
        - Ferreira RJ, Firmino PRA, Cristino CT (2015):
        A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes.
        PLoS ONE, 10(7), e0133772. https://doi.org/10.1371/journal.pone.0133772
        - Felix J, Firmino PRA, Ferreira RJ (2016):
        Kernel Density Estimation and Applications.
        Statistics, 50(3), 123-145. https://doi.org/10.1007/s00362-015-0704-5
        - Felix J, Firmino PRA, Ferreira RJ (2019):
        A Tool for Reliability Data Analysis.
        Applied Stochastic Models in Business and Industry, 35(4), 761-776. https://doi.org/10.1002/asmb.2396
        - Firmino PRA, Ferreira RJ (2021):
        Generalized Models in Reliability.
        Reliability Engineering & System Safety, 207, 107325. https://doi.org/10.1016/j.ress.2021.107325
    """

    try:
        if (x + v) > 0 and a > 0 and b > 0:
            result = (
                np.log(b)
                - b * np.log(a)
                + (b - 1) * np.log(x + v)
                + (v / a) ** b
                - ((x + v) / a) ** b
            )
        else:
            result = -np.inf
    except Exception as e:
        print(f'dwgrp_ERROR: {e}')
        result = -np.inf

    return result


def pwgrp(
    x,
    a: float,
    b: float,
    v: float,
    lower_tail: bool = True,
    log: bool = False,
) -> float:
    """
    WGRP Cumulative Distribution Function

    CDF of the WGRP (Weibull-based Generalized Renewal Process) distribution,
    with scale parameter `a`, shape parameter `b`, and virtual age `v`, at a given time set `x`.

    Parameters:
        x (list of float):
            The times at which the WGRP CDF must be computed. Values must be greater than 0.
        a (float):
            The scale parameter. Similarly to the Weibull distribution, one must work with `a > 0`.
        b (float):
            The shape parameter. Similarly to the Weibull distribution, one must work with `b > 0`.
            This parameter reflects the stage faced by the system under study, whether stable, improving or deteriorating.
            If `b = 1`, one has an exponential distribution underlying the process. It implies in a stable system, i.e.
            without memory (in which the intervention rate is constant through the time). In turn, if `b < 1`, one usually
            has an improving system (in which the intervention rate decreases as time evolves). Finally, if `b > 1`, one
            usually has a deteriorating system (in which the intervention rate increases as time evolves). However, the
            meaning of `b` might change depending on the value of the rejuvenation parameter (`q`).
        v (float):
            The virtual age of the system prior to `x`. For general purposes, `v` is computed mainly considering the
            rejuvenation parameter (`q`).
        lower_tail (bool):
            If True (default), probabilities are `P(X <= x)`. Otherwise, one has `P(X > x)`.
        log (bool):
            If True (default), the CDFs are given at log-scale.

    Returns:
        list of float The values of the WGRP CDF at `x`.

    Examples:
        >>> pwgrp(2, 1, 2, 1)
        0.9996645373720975
        >>> pwgrp(2, 1, 2, 1, lower_tail=False)
        0.00033546262790251185
        >>> pwgrp(2, 1, 2, 1, log=True)
        -0.0003355189080768247

    References:
        - Ferreira RJ, Firmino PRA, Cristino CT (2015):
        A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes.
        PLoS ONE, 10(7), e0133772. https://doi.org/10.1371/journal.pone.0133772
        - Felix J, Firmino PRA, Ferreira RJ (2016):
        Kernel Density Estimation and Applications.
        Statistics, 50(3), 123-145. https://doi.org/10.1007/s00362-015-0704-5
        - Felix J, Firmino PRA, Ferreira RJ (2019):
        A Tool for Reliability Data Analysis.
        Applied Stochastic Models in Business and Industry, 35(4), 761-776. https://doi.org/10.1002/asmb.2396
        - Firmino PRA, Ferreira RJ (2021):
        Generalized Models in Reliability.
        Reliability Engineering & System Safety, 207, 107325. https://doi.org/10.1016/j.ress.2021.107325
    """
    try:
        result = (v / a) ** b - ((v + x) / a) ** b

        if not log:
            result = np.exp(result)
            if lower_tail:
                result = 1 - result
        elif lower_tail:
            result = np.log(1 - np.exp(result))

    except Exception as e:
        print(f'pwgrp_ERROR: {e}')
        result = -1  # Retorna um valor padrão em caso de erro

    return float(result)


def lwgrp(x, a, b, q, propagations, log=True) -> float:
    """
    Calculate the log-likelihood function for the Weibull-based Generalized Renewal Process (WGRP) model.

    Parameters:
        x (array-like):
            Observed failure times.
        a (float):
            Parameter of the GRP model.
        b (float):
            Parameter of the GRP model.
        q (float):
            Parameter of the GRP model.
        propagations (int):
            Number of propagations for virtual age sampling.
        log (bool):
            If True returns the logarithm of the likelihood, if False returns the likelihood.

    Returns:
        res: float, the calculated (log) likelihood value.

    Examples:
        >>> x = [1, 2, 3, 4, 5]
        >>> a = 0.5
        >>> b = 1.5
        >>> q = 0.1
        >>> propagations = [1, 1, 0, 3, 2]
        >>> log = True
        >>> lwgrp(x, a, b, q, propagations, log)
        -83.60928510351891
    """
    n = len(x)
    virtualAges = get_sample_virtual_ages(x, q, propagations)
    previous_virtual_age = 0
    l = -np.inf

    try:
        if a > 0 and b > 0:
            l = 0
            for i in range(n):
                l += dwgrp(x[i], a, b, previous_virtual_age, log=True)
                previous_virtual_age = virtualAges[i]
    except Warning as war:
        print(f'lwgrp_WARNING: (a, b, q)= ({a}, {b}, {q}) {war}')
    except Exception as err:
        print(f'lwgrp_ERROR: (a, b, q)= ({a}, {b}, {q}) {err}')

    res = l
    if not log:
        res = np.exp(l)
    return float(res)
