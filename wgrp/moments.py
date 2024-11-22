import numpy as np
from scipy.special import gamma, gammaincc

from wgrp.virtual_ages import virtual_age
from wgrp.wgrp_functions import pwgrp


def _conditional_moments(parameters, v):
    a = parameters['a']
    b = parameters['b']
    x_aux = 1 / b
    q1 = (v / a) ** b
    q2 = 0

    def incomplete_gamma(x):
        integral_lower_bound = 1 / b
        return gammaincc(integral_lower_bound, x) * gamma(integral_lower_bound)

    upper_gamma1 = incomplete_gamma(q1)
    upper_gamma2 = incomplete_gamma(q2)
    gamma1 = gamma(1 + 1 / b)

    l_aux = (v / a) ** b
    l_aux += np.log(a) - np.log(b)
    l_aux2 = b * gamma1 + upper_gamma1 - upper_gamma2
    l_aux2 = np.log(l_aux2)
    l_aux += l_aux2

    aux = np.exp(l_aux) if np.isfinite(l_aux) and l_aux > 0 else 0

    Ex = aux if aux > 0 else 0
    Vx = 0

    # pwgrp necessita de revisão
    R_Ex = pwgrp(Ex, parameters['a'], parameters['b'], v, lower_tail=False)

    ret = {'mean': Ex, 'var': Vx, 'R_mean': R_Ex}
    return ret


def _sample_conditional_moments(parameters):
    n = parameters['nInterventions']
    q = parameters['q']
    Ex = np.zeros(n)
    Vx = np.zeros(n)
    R_Ex = np.zeros(n)
    virtualAges = np.zeros(n)

    current_virtual_age = parameters['previousVirtualAge']

    for i in range(n):
        cond_mom = _conditional_moments(parameters, current_virtual_age)
        Ex[i] = cond_mom['mean']
        Vx[i] = cond_mom['var']
        R_Ex[i] = cond_mom['R_mean']
        # verificar virtual_age
        virtual_age_result = virtual_age(
            parameters['propagations'][i], q, current_virtual_age, Ex[i]
        )
        current_virtual_age = virtual_age_result['virtualAge']
        virtualAges[i] = current_virtual_age

    ret = {'mean': Ex, 'var': Vx, 'R_mean': R_Ex, 'virtualAges': virtualAges}
    return ret
