import numpy as np
import pandas as pd


class Parameters:
    """
    The Parameters class encapsulates general parameters related to the WGRP 
    (Weibull-Based Generalized Renewal Processes). It also provides arguments for the 
     search for optimal parameters for the WGRP.

    Attributes:
    `INTERVENTION_TYPE` (dict): Types of intervention, including Preventive (PM) and Corrective (CM).
    PROPAGATION (dict): Types of propagation with values for Kijima I and Kijima II.
    PM_TIMES (str): String representing preventive maintenance times.
    TIMES_RELIABILITY (str): String representing the reliability of preventive maintenance times.
    PM_RELIABILITIES (str): String representing the reliability of preventive maintenances.
    nearZero (float): A value very close to zero to avoid numerical problems.
    boundsNames (list): List containing the names of the minimum and maximum bounds of the 
    WGRP parameters space.
    aBounds (dict): Dictionary containing the minimum and maximum bounds for the WGRP scale 
    parameter 'a'.
    bBounds (dict): Dictionary containing the minimum and maximum bounds for the WGRP shape 
    parameter 'b'.
    qBounds (dict): Dictionary containing the minimum and maximum bounds for the WGRP 
    rejuvenation parameter 'q'.
    FORMALISM (dict): Different formalisms and types of propagation for the WGRP.
    """

    def __init__(self):
        self.INTERVENTION_TYPE = {'PM': 'Preventive', 'CM': 'Corrective'}
        self.PROPAGATION = {'KijimaI': 1, 'KijimaII': 0}

        self.PM_TIMES = 'PM TIMES'
        self.TIMES_RELIABILITY = 'PM TIMES RELIABILITY'
        self.PM_RELIABILITIES = 'PMs RELIABILITIES'

        self.nearZero = 1e-100
        self.boundsNames = ['min', 'max']

        a_bounds_list = [self.nearZero, -1]
        b_bounds_list = [self.nearZero, 5]
        q_bounds_list = [0, 1]

        self.aBounds = dict(zip(self.boundsNames, a_bounds_list))
        self.bBounds = dict(zip(self.boundsNames, b_bounds_list))
        self.qBounds = dict(zip(self.boundsNames, q_bounds_list))

        self.FORMALISM = {
            'RP': 'RP',
            'NHPP': 'NHPP',
            'KIJIMA_I': 'Kijima I',
            'KIJIMA_II': 'Kijima II',
            'INTERVENTION_TYPE': 'Intervention type-based',
            'GENERIC_PROPAGATION': 'Generic propagation-based',
        }


class Get:
    """
    Class `Get`

    The `Get` class contains functions that fetch immediate return values without requiring 
    computational processing.
    It serves to retrieve parameters and other immediate data, leveraging the 
    `Parameters` class for default settings and bounds in specific contexts.
    """

    def __init__(self):
        self.parameters = Parameters()

    def get_parameters(
        self,
        nSamples: int = 0,
        nInterventions: int = None,
        a: float = None,
        b: float = None,
        q: float = 0.5,
        propagations: list = None,
        reliabilities: list = None,
        previousVirtualAge: int = 0,
        interventionsTypes: list = None,
        formalism: str = 'RP',
        cumulativeFailureCount: int = None,
        timesPredictFailures: list = None,
        nIntervetionsReal: int = None,
    ) -> dict:
        """
        Function `get_parameters`

        The `get_parameters` method encapsulates methods for retrieving parameters related to 
        the work group model (`wgrp`).
        It utilizes an instance of the `Parameters` class to access default settings and 
        bounds for function minimization
        in the search for optimal parameters for the `wgrp`.

        Args:
            nSamples (int): Number of samples to be simulated in the inference process.
            nInterventions (int): Number of interventions.praf: parei
            a (float): Parameter a.
            b (float): Parameter b.
            q (float): Parameter q.
            propagations (list): List of propagations.
            reliabilities (list): List of reliabilities.
            previousVirtualAge (int): Previous virtual age.
            interventionsTypes (list): List of intervention types.
            formalism (str): Formalism type.
            cumulativeFailureCount (int): Cumulative failure count.
            timesPredictFailures (list): Times to predict failures.
            nIntervetionsReal (int): Number of real interventions.

        Returns:
            dict: A dictionary containing the parameters and their values.

        Examples:
            >>> params = Get().get_parameters(nSamples=10, a=0.1, b=0.2, q=0.6, formalism='RP')
            >>> params['nSamples']
            10
            >>> params['a']
            0.1
            >>> params['b']
            0.2
            >>> params['q']
            0.6
            >>> params['formalism']
            'RP'
            >>> params['bBounds']
            {'min': 1e-100, 'max': 5}
            >>> params['qBounds']
            {'min': 0, 'max': 1}
        """
        parameters = {
            'nSamples': nSamples,
            'nInterventions': nInterventions,
            'a': a,
            'b': b,
            'q': q,
            'propagations': propagations,
            'reliabilities': reliabilities,
            'previousVirtualAge': previousVirtualAge,
            'interventionsTypes': interventionsTypes,
            'formalism': formalism,
            'cumulativeFailureCount': cumulativeFailureCount,
            'timesPredictFailures': timesPredictFailures,
            'nIntervetionsReal': nIntervetionsReal,
            'bBounds': self.parameters.bBounds,  # Accesses the bounds for 'b' from Parameters
            'qBounds': self.parameters.qBounds,  # Accesses the bounds for 'q' from Parameters
        }

        return parameters

    @staticmethod
    def get_optimum(mle_objs, df) -> dict:
        """
        Return the best model found based on the minimum BIC score.

        Args:
            mle_objs (list or dict): List or dictionary of maximum likelihood estimation objects.
            df (pandas.DataFrame): DataFrame containing the BIC scores and formalism information.

        Returns:
            object: The maximum likelihood estimation object corresponding to the model with the lowest BIC score.

        Raises:
            ValueError: If `df` is empty or if the indices do not align properly.

        Examples:
            >>> mle_objs = [('model1'), ('model2'), ('model3')]
            >>> data = {'BIC': [100.5, 95.3, 110.2], 'Formalism': ['formalism1', 'formalism2', 'formalism3']}
            >>> df = pd.DataFrame(data)
            >>> best_model = Get().get_optimum(mle_objs, df)
            >>> best_model
            'model2'
        """

        # Identify the formalism of the model with the minimum BIC score
        idx_min_bic = df['BIC'].idxmin()
        best_formalism = df.loc[idx_min_bic, 'Formalism']

        # Generate a string identifier for the optimum model
        str_optimum = f'optimum_{best_formalism}'

        return mle_objs[idx_min_bic]