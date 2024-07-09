from mwgrp.base_functions import Get, Parameters
from mwgrp.mle_wgrp import MleWgrp


def getMLE_objs(timesBetweenInterventions, interventionsTypes, b=1):
    get_parameters = Get().get_parameters
    FORMALISM = Parameters().FORMALISM
    PROPAGATION = Parameters().PROPAGATION
    mle_wrgp = MleWgrp
    x = timesBetweenInterventions
    j = interventionsTypes

    if b != 1:
        b = b

    parameters_RP = get_parameters(b=b, formalism=FORMALISM['RP'])
    # print(parameters_RP)
    optimum_RP = mle_wrgp(x=x, p_parameters=parameters_RP).minimization()

    parameters_NHPP = get_parameters(b=b, formalism=FORMALISM['NHPP'])
    optimum_NHPP = mle_wrgp(x, p_parameters=parameters_NHPP).minimization()

    best = {'b': optimum_RP['b'], 'q': 0}
    if optimum_RP['optimum'][0] < optimum_NHPP['optimum'][0]:
        best['b'] = optimum_NHPP['b']
        best['q'] = 1
    # print(optimum_NHPP)
    parameters_KijimaI = get_parameters(
        b=best['b'], q=best['q'], formalism=FORMALISM['KIJIMA_I']
    )
    optimum_KijimaI = mle_wrgp(
        x, p_parameters=parameters_KijimaI
    ).minimization()

    parameters_KijimaII = get_parameters(
        b=best['b'], q=best['q'], formalism=FORMALISM['KIJIMA_II']
    )
    optimum_KijimaII = mle_wrgp(
        x, p_parameters=parameters_KijimaII
    ).minimization()

    if optimum_KijimaI['optimum'][0] < optimum_KijimaII['optimum'][0]:
        best['b'] = optimum_KijimaII['b']
        best['q'] = optimum_KijimaII['q']

    parameters_InterventionType = get_parameters(
        b=best['b'],
        q=best['q'],
        interventionsTypes=j,
        propagations=[PROPAGATION['KijimaI'], PROPAGATION['KijimaII']],
        formalism=FORMALISM['INTERVENTION_TYPE'],
    )
    optimum_InterventionType = mle_wrgp(
        x, p_parameters=parameters_InterventionType
    ).minimization()

    # mle_objs = [
    #     {'optimum_RP': optimum_RP},
    #     {'optimum_NHPP': optimum_NHPP},
    #     {'optimum_KijimaI': optimum_KijimaI},
    #     {'optimum_KijimaII': optimum_KijimaII},
    #     {'optimum_InterventionType': optimum_InterventionType}
    # ]
    mle_objs = [
        optimum_RP,
        optimum_NHPP,
        optimum_KijimaI,
        optimum_KijimaII,
        optimum_InterventionType,
    ]

    return mle_objs
