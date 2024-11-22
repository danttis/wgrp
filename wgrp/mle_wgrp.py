import os
import sys
from contextlib import contextmanager

import numpy as np
from pyswarm import pso

from wgrp.base_functions import *
from wgrp.virtual_ages import _get_virtual_ages_and_a
from wgrp.wgrp_functions import lwgrp

from scipy import optimize

class MleWgrp:

    """
    Maximum Likelihood Estimation (MLE) process to find the parameters `a`, `b`, and `q` used in the WGRP model.
    This class gathers a set of functions and configurations to calculate the parameters used in WGRP using analytical or probabilistic methods.

    Parameters:
        x (array-like):
            Input data set representing failure times or events.

        p_parameters (dict):
            Dictionary containing the parameters necessary for modeling and optimization:
                - `formalism`: String indicating the formalism to be used ('RP', 'NHPP', 'KIJIMA I', 'Kijima II', 'Intervention type-based', 'Generic propagation-based').
                - `interventionsTypes`: String indicating the type of interventions ('PM', 'CM').
                - `bBounds`: dict containing 'min' and 'max', lower and upper bounds for the parameter 'b'.
                - `qBounds`: dict containing 'min' and 'max', lower and upper bounds for the parameter 'q'.

    References:
        - Ferreira RJ, Firmino PRA, Cristino CT (2015):
        A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes.
        PLoS ONE, 10(7), e0133772. https://doi.org/10.1371/journal.pone.0133772
    """

    def __init__(self, x, p_parameters, random_state=0, optimizer="ps"):
        self.x = x
        self.p_parameters = p_parameters
        self.random_state = random_state
        self.optimizer = optimizer
        self.n = len(x)
        np.random.seed(self.random_state)
        self.FORMALISM = Parameters().FORMALISM
        self.PROPAGATION = Parameters().PROPAGATION
        self.INTERVENTION_TYPE = Parameters().INTERVENTION_TYPE

    def objective_function(self, parameters):
        b = parameters[0]
        propagations = np.zeros(self.n, dtype=int)
        q = np.inf
        if self.p_parameters['formalism'] == self.FORMALISM['RP']:
            q = 0
        elif self.p_parameters['formalism'] == self.FORMALISM['NHPP']:
            q = 1
        elif self.p_parameters['formalism'] == self.FORMALISM['KIJIMA_I']:
            q = parameters[1]
            propagations = np.full(self.n, self.PROPAGATION['KijimaI'])
        elif self.p_parameters['formalism'] == self.FORMALISM['KIJIMA_II']:
            q = parameters[1]
            propagations = np.full(self.n, self.PROPAGATION['KijimaII'])
        elif (
            self.p_parameters['formalism']
            == self.FORMALISM['INTERVENTION_TYPE']
        ):
            q = parameters[1]
            intervention_type = self.p_parameters['interventionsTypes']
            if intervention_type == self.INTERVENTION_TYPE['PM']:
                propagations = np.full(self.n, parameters[2])
            elif intervention_type == self.INTERVENTION_TYPE['CM']:
                propagations = np.full(self.n, parameters[3])
        elif (
            self.p_parameters['formalism']
            == self.FORMALISM['GENERIC_PROPAGATION']
        ):
            q = parameters[1]
            propagations = parameters[2 : (self.n + 2)]

        log_likelihood = -np.inf

        if b > 0:
            virtual_a = _get_virtual_ages_and_a(b, q, propagations, self.x)['a']
            if np.isfinite(virtual_a):
                log_likelihood = lwgrp(
                    self.x, virtual_a, b, q, propagations, log=True
                )

        return -log_likelihood  # Negative for minimization

    def minimization(self):
        """
        The `minimization` function calculates parameters and is the most significant function in the class for end users.
        It employs a particle swarm optimization (PSO) method to compute the parameters `b` and `q`, which cannot be analytically
        determined. This function does not take any parameters directly as it encapsulates a full execution of the `MleWgrp` class.

        Returns:
            (dict): A dictionary containing the following key-value pairs:
                - 'a': The calculated parameter `a`.
                - 'b': The optimized value of parameter `b`.
                - 'q': The optimized value of parameter `q`.
                - 'propagations': Optional, propagation values depending on the formalism.
                - 'virtualAges': The computed virtual ages.
                - 'optimum': The optimal values found by the optimization process.
                - 'parameters': The input parameters `p_parameters` used for the optimization.
        """
        b = 1
        q = 0
        lower = [self.p_parameters['bBounds']['min']]   # Lower limit for 'b'
        upper = [self.p_parameters['bBounds']['max']]  # Upper limit for 'b'
        par = [b]  # Initial parameter 'b'

        # Conditions based on 'p_parameters['formalism']'
        if self.p_parameters['formalism'] == self.FORMALISM['RP']:
            q = 0
        elif self.p_parameters['formalism'] == self.FORMALISM['NHPP']:
            q = 1
        else:
            lower.extend(
                [self.p_parameters['qBounds']['min']]
            )  # Lower limit for 'q'
            upper.extend(
                [self.p_parameters['qBounds']['max']]
            )  # # Upper limit for  'q'
            par.extend(
                [self.p_parameters['q']]
            )   # Adding 'q' to initial parameters
            if (
                self.p_parameters['formalism']
                == self.FORMALISM['INTERVENTION_TYPE']
            ):
                lower.extend([0, 0])  # Lower limits for type intervention
                upper.extend([1, 1])  # Upper limits for type intervention
            elif (
                self.p_parameters['formalism']
                == self.FORMALISM['GENERIC_PROPAGATION']
            ):
                lower.extend(
                    [0] * self.n
                )  # Lower bounds for generic propagation
                upper.extend(
                    [1] * self.n
                )  # Upper limits for generic propagation

        # Control configuration for optimization (PSO)
        options = {'maxiter': 1000, 'debug': False}

        @contextmanager
        def suppress_stdout():
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
  
        
        bounds = list(zip(lower, upper))

        if self.optimizer == "ps":
            with suppress_stdout():
                optimum, value = pso(
                    MleWgrp(self.x, self.p_parameters, self.random_state).objective_function,
                    lower,
                    upper,
                    swarmsize=1000,
                    args=(),
                    **options
                )
            b = (
            optimum[0] if len(optimum) > 0 else self.p_parameters['b']
            )  # If there is no optimal value, use the initial one
            optimum_value = -value
            
        else: 
            with suppress_stdout():
                result = optimize.dual_annealing(
                        func=MleWgrp(self.x, self.p_parameters, self.random_state).objective_function,
                        bounds=bounds,
                        maxiter=10000,          
                        initial_temp=40,       
                        maxfun=5000,            
                        no_local_search=False,  
                        minimizer_kwargs={'method': 'L-BFGS-B'} 
                    )

            optimum = result.x 
            b = (
                optimum[0] if len(optimum) > 0 else self.p_parameters['b']
            ) 
            optimum_value = -result.fun  # valor mínimo da função objetivo
                # (
        #     -optimum[0] if len(optimum) > 0 else -np.inf
        # )  # Log-likelihood value (or -inf if not optimized)
        propagations = None

        if len(optimum) > 1:
            if (
                self.p_parameters['formalism'] != self.FORMALISM['RP']
                and self.p_parameters['formalism'] != self.FORMALISM['NHPP']
            ):
                q = optimum[1]
                if (
                    self.p_parameters['formalism']
                    == self.FORMALISM['KIJIMA_I']
                ):
                    propagations = np.full(self.n, self.PROPAGATION['KijimaI'])
                elif (
                    self.p_parameters['formalism']
                    == self.FORMALISM['KIJIMA_II']
                ):
                    propagations = np.full(
                        self.n, self.PROPAGATION['KijimaII']
                    )
                elif (
                    self.p_parameters['formalism']
                    == self.FORMALISM['INTERVENTION_TYPE']
                ):
                    if (
                        self.p_parameters['interventionsTypes']
                        == self.INTERVENTION_TYPE['PM']
                    ):
                        propagations = np.full(self.n, optimum[2])
                    elif (
                        self.p_parameters['interventionsTypes']
                        == self.INTERVENTION_TYPE['CM']
                    ):
                        propagations = np.full(self.n, optimum[3])
                elif (
                    self.p_parameters['formalism']
                    == self.FORMALISM['GENERIC_PROPAGATION']
                ):
                    propagations = optimum[2 : (self.n + 2)]

        # Checking and adjusting "a" and "virtualAges"
        a_vs = _get_virtual_ages_and_a(b, q, propagations, self.x)
        a = a_vs['a']
        virtualAges = a_vs['virtualAges']

        # Update optimal parameters found
        self.p_parameters['a'] = a
        self.p_parameters['b'] = b
        self.p_parameters['q'] = q
        self.p_parameters['propagations'] = propagations

        optimal_parameters = {
            'a': a,
            'b': b,
            'q': q,
            'propagations': propagations,
            'virtualAges': virtualAges,
            'optimum': optimum,
            'parameters': self.p_parameters,
            'optimum_value': optimum_value,
        }

        return optimal_parameters
