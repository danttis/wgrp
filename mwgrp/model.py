import numpy as np

from mwgrp.compute import *
from mwgrp.getcomputer import *
from mwgrp.moments import sample_conditional_moments


def fit_f(data, type='date', time_unit='days', accumulated=False):
    if type not in ['date', 'numeric']:
        raise ValueError("Invalid type. Expected 'date' or 'numeric'.")

    np.random.seed(0)
    if type == 'date':
        newData_i = append_times_between_events_from_date_events(
            data, time_unit=time_unit
        )
        TBEs = newData_i['TBE']
        event_types = list(newData_i['event_type'])
        
    elif type == 'numeric':
        if accumulated:
            accumulated_time = [
                data[i + 1] - data[i] for i in range(len(data) - 1)
            ]
            TBEs = accumulated_time
        else:
            TBEs = data
        event_types = ['Corrective'] * len(TBEs)
      
    mle_objs = getMLE_objs(
        timesBetweenInterventions=list(TBEs),
        interventionsTypes=event_types,
        b=1,
    )
    mle_objs = mle_objs

    return mle_objs, TBEs


PROPAGATION = Parameters().PROPAGATION
get_parameters = Get().get_parameters
get_optim = Get().get_optimum


def pred(qtd, mle_objs, TBEs, quantile=0.2, events_in_the_future_tense = 0):
    df = summarize_ics_and_parameters_table(mle_objs, TBEs)[
        'df1'
    ]
    nEventsAheadToPredict = qtd
    # probabilityOfFailure = 5   # revisar
    # globals()['quantile']
    n = len(TBEs)

    optimum = get_optim(mle_objs, df)
    # print(optimum)
    print(f"alpha = {optimum['a']}")
    print(f"beta = {optimum['b']}")
    print(f"q = {optimum['q']}")
    # Verifica se propagations é nulo (RP ou NHPP).
    if optimum['propagations'] is None:
        optimum['propagations'] = np.ones(n)

    cF = np.sum(TBEs)
    tPredictFailures =  events_in_the_future_tense  # globals()['failuresInTime']
    m = nEventsAheadToPredict

    pmPropagations = np.concatenate(
        (optimum['propagations'], np.repeat(PROPAGATION['KijimaII'], m))
    )
    parameters = get_parameters(
        nSamples=1000,
        nInterventions=(n + m),
        a=optimum['a'],
        b=optimum['b'],
        q=optimum['q'],
        propagations=pmPropagations,
        cumulativeFailureCount=cF,
        timesPredictFailures=tPredictFailures,
        nIntervetionsReal=n,
    )
   
    bSample = bootstrap_sample(parameters)
    theoreticalMoments = sample_conditional_moments(parameters)

    forecasting = cumulative_forecast_times(
        x=TBEs,
        bootstrap_sample=bSample,
        conditional_means=theoreticalMoments,
        parameters=parameters,
        probability_of_failure=qtd,  # revisar
        quantile=quantile
    )   # x=None, bootstrap_sample=None, conditional_means=None, parameters=None, probability_of_failure=0, quantile=0.1

    forecasting_final = compute_forecasting_table(forecasting, initial_time=10)

    return forecasting_final, optimum, df, parameters


class wgrp_model:
    """The `Wgrp_model` class is the main function of the package and controls all other functions.
    Although all other functions can be used separately, this class provides two main functions:

    - `fit`: Works similarly to many machine learning packages, fitting the model to the data and returning a dataframe and a list with the parameters of each formalism and a list with the time between the given dates.
    - `predict`: Also works similarly to machine learning packages, receiving the number of events to be calculated in the future and returning a dataframe with four columns: one for the index of each event, one for the lower quartile of 2.5%, one for the upper quartile of 95%, and the mean of the quartiles, which is the actual forecast.
    """

    def __init__(self):
        self.TBEs_ = None    # Attributes to store the fitting results
        self.name = None
        self.mle_objs_ = None
        self.optimum_ = None
        self.df_ = None
        self.quantile_s = None
        self.quantile_i = None
        self.quantile_n = None
        self.parameters = None
        self.events_in_the_future_tense = None

    def fit(self, data, type='date', time_unit='days', accumulated=False):
        """
        Fits the model to the provided data. Although the function does not return anything explicitly, it is still possible to access `mle_objs_`, an object containing a list of Maximum Likelihood Estimation (MLE) objects, and `TBEs_`, a list of times between events (TBEs).

        Parameters:
            data (pd.DataFrame or list):
                Data to be fitted by the model. If it contains intervention dates, they should be in a dataframe with columns `date` and `event_type`, where interventions are of types `Preventive` or `Corrective`. If only the times between events of interest are provided, pass a list and change the argument `type` to `numeric`.
            type (str):
                Type of the provided data. If `type='date'`, the object should be a dataframe with a column containing event dates and another column with intervention types. If `type='numeric'`, provide only a list with times between events of interest. If the type is `numeric`, there are two options: if the time is between events, for example: [2, 4, 3, 5], or if the time is accumulated, for example: [2, 6, 9, 14]. If it is accumulated, pass `True` to the `accumulated` parameter.
            time_unit (str):
                Time unit for analyzing intervals between interventions. Can be 'weeks', 'days', 'hours', 'minutes', 'seconds', 'microseconds', 'milliseconds'. Default is 'days'.
            accumulated (bool):
                Indicates if the provided times are accumulated. If `True`, the times are considered accumulated. Default is `False`.

        Examples:
            >>> failures = [0.2, 1, 5, 7, 89, 21, 12]
            >>> model = wgrp_model()
            >>> model.fit(failures, type='numeric', time_unit='minutes')
            >>> model.mle_objs_[0]
            {'a': np.float64(13.449147109006473), 'b': np.float64(0.6284720253731791), 'q': 0, 'propagations': None, 'virtualAges': [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)], 'optimum': array([0.62847203]), 'parameters': {'nSamples': 0, 'nInterventions': None, 'a': np.float64(13.449147109006473), 'b': np.float64(0.6284720253731791), 'q': 0, 'propagations': None, 'reliabilities': None, 'previousVirtualAge': 0, 'interventionsTypes': None, 'formalism': 'RP', 'cumulativeFailureCount': None, 'timesPredictFailures': None, 'nIntervetionsReal': None, 'bBounds': {'min': 1e-100, 'max': 5}, 'qBounds': {'min': 0, 'max': 1}}, 'optimum_value': -26.12961862148702}
        """
        # Calls the fit_f method of Fit_grp to fit the model
        self.mle_objs_, self.TBEs_ = fit_f(data, type, time_unit, accumulated)

    def predict(self, qtd=1, quantile=0.2, events_in_the_future_tense=0):
        """
        Makes future predictions based on the desired number of events.

        self.optimum_: This attribute stores the optimum value calculated during the prediction process. It is updated with each call to the predict function.
        self.df_: This attribute stores the DataFrame used in the prediction calculations. It is updated with new predictions each time the predict function is called.

        Parameters:
            qtd (int): Number of future events to be calculated.

        Returns:
            (DataFrame): DataFrame containing the predictions with lower quartile (2.5%), upper quartile (95%), and the mean of the quartiles.

        Examples:
            >>> failures = [0.2, 1, 5, 7, 89, 21, 12]
            >>> model = wgrp_model()
            >>> model.fit(failures, type='numeric', time_unit='minutes')
            >>> predictions = model.predict(3)
            alpha = 1.1910974925051054
            beta = 0.41123404255463386
            q = 1
        """
        predictions, self.optimum_, self.df_, self.parameters = pred(
            qtd, list(self.mle_objs_), list(self.TBEs_), quantile, events_in_the_future_tense
        )
        self.quantile_s, self.quantile_i, self.quantile_n, self.events_in_the_future_tense = predictions['dataframe']['Quantile_97.5'], predictions['dataframe']['Quantile_2.5'], predictions['dataframe']['newQuantile'], predictions['qtd_events']
        return predictions['dataframe'][['Intervention', 'Mean']]
