import numpy as np

from wgrp.compute import *
from wgrp.getcomputer import *
from wgrp.moments import sample_conditional_moments


def _fit(data, type='date', time_unit='days', cumulative=False):
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
        if cumulative:
            cumulative_time = [
                data[i + 1] - data[i] for i in range(len(data) - 1)
            ]
            TBEs = cumulative_time
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


def _pred(qtd, mle_objs, TBEs, events_in_the_future_tense = 0, best_prediction=False):
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
    
    if best_prediction:
        nSamples = 100000
    else: 
        nSamples=1000

    pmPropagations = np.concatenate(
        (optimum['propagations'], np.repeat(PROPAGATION['KijimaII'], m))
    )
    parameters = get_parameters(
        nSamples=nSamples,
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
        probability_of_failure=qtd,
        best_prediction=best_prediction
    )   # x=None, bootstrap_sample=None, conditional_means=None, parameters=None, probability_of_failure=0, quantile=0.1

    forecasting_final = compute_forecasting_table(forecasting, initial_time=10)

    return forecasting_final, optimum, df, parameters


class wgrp_model:
    """The `wgrp_model` class is the main function of the package and it controls all other functions.
    Although all other functions can be used separately, this class provides two main functions:

    - `fit`: Works similarly to many machine learning packages, fitting WGRP models to the times between 
    events (TBEs) or times to occur events (TTOs) data and returning a DataFrame with the parameters of a 
    number of WGRP formalisms (i.e. Renew Processes - RP, Non-Homogeneous Poisson Processes - NHPP, Kijima I, 
    Kijima II, and Intervention type-based models). Further, a list with the TBEs is returned.
    - `predict`: Also works similarly to machine learning packages, receiving the number of events for 
    which the times until occurrence must be forecasted (i.e. out-of-sample predictions) and returning a 
    DataFrame with four columns: the index of each event, the 2.5% quantile (i.e. the lower bound of the
      95% confidence interval), the 97.5% quantile (i.e. the upper bound of the 95% confidence interval), 
      and the mean value of the times to occur the events under study.
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
        self.best_prediction = None

    def fit(self, data, type='date', time_unit='days', cumulative=False):
        """
        Fits WGRP models to the provided data. Although the function does not return anything explicitly, 
        it computes the `mle_objs_` attribute, a list of Maximum Likelihood Estimation (MLE) objects, and 
        `TBEs_`, a list of times between events (TBEs).

        Parameters:
            data (pd.DataFrame or list):
                Data to be fitted by the model.One can use a DataFrame with columns named `date` and 
                `event_type` (assuming values like `Preventive` or `Corrective`, for instance). One can 
                also use a list with the TBEs (`numeric` values); in this case, the nature of the interventions is not
                taken into account.
            type (str):
                Type of the provided data.  Default is `date`. If `type = date`, the `data` object should be a DataFrame 
                with a column named `date` containing event dates and other column named `event_type` with the respective
                  intervention types. If `type = numeric`, the `data` object should be a list of TBEs or times to occur the
                  events (TTOs); in this case, the nature of the interventions is not taken into account. If 
                  `type = numeric`, there are two options for `data": TBEs if `cumulative = False` (e.g. 
                  [2, 4, 3, 5]), or TTOs if `cumulative = True` (e.g. [2, 6, 9, 14]).
            time_unit (str):
                Time unit for analyzing intervals between interventions. It can be 'weeks', 'days', 'hours', 
                'minutes', 'seconds', 'microseconds', 'milliseconds'. Default is 'days'.
            cumulative (bool):
                Indicates if the provided numeric times are cumulative. Default is `False`. there are two 
                options for `data": TBEs if `cumulative = False` (e.g. [2, 4, 3, 5]), or TTOs if 
                `cumulative = True` (e.g. [2, 6, 9, 14]). 

        Examples:
            >>> TBEs = [0.2, 1, 5, 7, 89, 21, 12]
            >>> model = wgrp_model()
            >>> model.fit(TBEs, type='numeric', time_unit='minutes')
            >>> model.mle_objs_[0]
            {'a': np.float64(13.449147109006473), 'b': np.float64(0.6284720253731791), 'q': 0, 'propagations': None, 'virtualAges': [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)], 'optimum': array([0.62847203]), 'parameters': {'nSamples': 0, 'nInterventions': None, 'a': np.float64(13.449147109006473), 'b': np.float64(0.6284720253731791), 'q': 0, 'propagations': None, 'reliabilities': None, 'previousVirtualAge': 0, 'interventionsTypes': None, 'formalism': 'RP', 'cumulativeFailureCount': None, 'timesPredictFailures': None, 'nIntervetionsReal': None, 'bBounds': {'min': 1e-100, 'max': 5}, 'qBounds': {'min': 0, 'max': 1}}, 'optimum_value': -26.12961862148702}
        """
        # Calls the fit_f method of Fit_grp to fit the model
        self.mle_objs_, self.TBEs_ = _fit(data, type, time_unit, cumulative)

    def predict(self, qtd=1, events_in_the_future_tense=0, best_prediction=False):
            """
            Makes future (out-of-sample) forecasts based on the desired number of steps ahead.

            Attributes:
                self.optimum_: Stores the optimum value calculated during the prediction process. It is 
                updated with each call of the predict function.
                self.df_: Stores the DataFrame used in the prediction calculations. It is updated with new 
                predictions each time the predict function is called.

            Parameters:
                qtd (int): Number of future events to be calculated.
                events_in_the_future_tense (int, optional): Number of events to be considered in the future. 
                Default is 0.
                best_prediction (bool, optional): If True, the method generates 100,000 random series and 
                selects the series with the lowest MSE, returning an average combination of these series. 
                This process is much slower than the original modeling. If False, it returns the best quantile.

            Returns:
                DataFrame: DataFrame containing predictions with the lower quartile (2.5%), upper quartile (97.5%), 
                and the mean TTOs.

            Examples:
                >>> TBEs = [0.2, 1, 5, 7, 89, 21, 12]
                >>> model = wgrp_model()
                >>> model.fit(TBEs, type='numeric', time_unit='minutes')
                >>> predictions = model.predict(3)
                alpha = 1.1910974925051054
                beta = 0.41123404255463386
                q = 1
            """

            predictions, self.optimum_, self.df_, self.parameters = _pred(
                qtd, list(self.mle_objs_), list(self.TBEs_), events_in_the_future_tense, best_prediction
            )
            self.quantile_s, self.quantile_i, self.quantile_n, self.events_in_the_future_tense, self.best_prediction = predictions['dataframe']['Quantile_97.5'], predictions['dataframe']['Quantile_2.5'], predictions['dataframe']['newQuantile'], predictions['qtd_events'], predictions['dataframe']['best_prediction']
            return predictions['dataframe'][['Intervention', 'Mean']]
