from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from wgrp.base_functions import Parameters
from wgrp.wgrp_functions import ic_wgrp, qwgrp

FORMALISM = Parameters().FORMALISM

def _root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def bootstrap_sample(parameters):
    """
    Generates random time series of failures using the wgrp model parameters.

    This function uses the parameters alpha, beta, and q from the wgrp model to simulate failure series. 
    For each sample, the `qwgrp` function is called to calculate the failure times, which are stored in a sample matrix.

    Parameters (dict):
        Dictionary containing the following parameters required for series generation:
            nSamples (int): number of samples to be generated.
            nInterventions (int): number of interventions in each sample.
            a (float): value of the alpha parameter for the wgrp model.
            b (float): value of the beta parameter for the wgrp model.
            q (float): value of the q parameter for the wgrp model.
            propagations (int) or (list): number of propagations to be performed.
            previousVirtualAge (int): initial previous virtual age of accumulated failures.
            cumulativeFailureCount (int): cumulative count of failures.
            timesPredictFailures (float): prediction time for future failures.

    Returns(dict):
        Dictionary with the following keys:
        sample_matrix: matrix (numpy array) with failure times for each sample, where each row represents a sample
          and each column represents an intervention.
        events_in_the_future_tense: list of mean times for predicted failures.

    Example:
    --------
    ```
    parameters = {
        'nSamples': 100,
        'nInterventions': 10,
        'a': 0.5,
        'b': 1.5,
        'q': 0.2,
        'propagations': 5,
        'previousVirtualAge': 10,
        'cumulativeFailureCount': 0,
        'timesPredictFailures': 20
    }
    result = bootstrap_sample(parameters)
    ```
    References:
    - Ferreira RJ, Firmino PRA, Cristino CT (2015):
    A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes.
    PLoS ONE, 10(7), e0133772. https://doi.org/10.1371/journal.pone.0133772
    """
    n_samples = parameters['nSamples']
    n_interventions = parameters['nInterventions']
    sample_matrix = np.zeros((n_samples, n_interventions))
    

    for i in range(0, n_samples):
        sample = qwgrp(
            n=n_interventions,
            a=parameters['a'],
            b=parameters['b'],
            q=parameters['q'],
            propagations=parameters['propagations'],
            reliabilities=None,
            failures_predict_count=True,
            previous_virtual_age=parameters['previousVirtualAge'],
            cumulative_failure_count=parameters['cumulativeFailureCount'],
            times_predict_failures=parameters['timesPredictFailures']
        )

        sample_matrix[i, :] = sample['times']
        times_predict_failures = sample['timesFailutesMeans']

    return {'sample_matrix': sample_matrix, 'events_in_the_future_tense': times_predict_failures}


def accumulate_values(sequence):
    """
    Computes the cumulative sum of a sequence of numbers.

    Parameters:
    sequence (iterable): A sequence (e.g., list, tuple) of numerical values to be accumulated.

    Returns:
    list: A list containing the accumulated sums. Each element corresponds to the sum of the values in the sequence up to that index.

    Example:
    >>> accumulate_values([1, 2, 3, 4])
    [1, 3, 6, 10]
    """
    accumulated = 0
    accumulated_values = []

    for value in sequence:
        accumulated += value
        accumulated_values.append(accumulated)

    return accumulated_values


def _cumulative_forecast_times(
    x=None,
    bootstrap_sample=None,
    conditional_means=None,
    parameters=None,
    probability_of_failure=0,
    top_n_series=3
):
    quantiles = [0.975, 0.025]
    percentis = [i/100 for i in range(1, 100)]
    quantiles.extend(percentis)

    cum_x = None
    lwd = {}
    lty = {}
    rq_x = None
    cum_q = None

    if probability_of_failure:
        n_probability_of_the_not_failing = (
            1
            - qwgrp(
                int(probability_of_failure),
                parameters['a'],
                parameters['b'],
                parameters['q'],
                parameters['propagations'],
            )['times']
        )
    else:
        n_probability_of_the_not_failing = 0

    if parameters['reliabilities'] is not None:
        rq_x = qwgrp(
            parameters['parameters']['nInterventions'],
            parameters['a'],
            parameters['b'],
            parameters['q'],
            parameters['propagations'],
            parameters['parameters']['reliabilities'],
        )['times']
        # print("passou")
        lwd['quantile'] = 4
        lty['quantile'] = 2
        cum_q = np.zeros(len(parameters['reliabilities']))
        cum_q[0] = rq_x[0]
        for i in range(1, len(parameters['reliabilities'])):
            cum_q[i] = cum_q[i - 1] + rq_x[i]

    n_cm = len(conditional_means['mean']) if conditional_means else 0
    cum_cm = None
    ql = None
    qu = None
    qc = None
    qn = None

    if n_cm > 0:   # aqui
        cum_cm = np.zeros(n_cm)
        ql = np.zeros(n_cm)
        qu = np.zeros(n_cm)
        qc = np.zeros(n_cm)
        qn = np.zeros(n_cm)

        cum_cm[0] = conditional_means['mean'][0]
        ql[0] = qwgrp(
            1,
            parameters['a'],
            parameters['b'],
            parameters['q'],
            [parameters['propagations'][0]],
            reliabilities=[quantiles[0]],
            previous_virtual_age=parameters['previousVirtualAge'],
        )['times'][0]
        qu[0] = qwgrp(
            1,
            parameters['a'],
            parameters['b'],
            parameters['q'],
            [parameters['propagations'][0]],
            reliabilities=[quantiles[1]],
            previous_virtual_age=parameters['previousVirtualAge'],
        )['times'][0]
        qc[0] = qwgrp(
            1,
            parameters['a'],
            parameters['b'],
            parameters['q'],
            [parameters['propagations'][0]],
            reliabilities=[quantiles[2]],
            previous_virtual_age=parameters['previousVirtualAge'],
        )['times'][0]
        qn[0] = qwgrp(
            1,
            parameters['a'],
            parameters['b'],
            parameters['q'],
            [parameters['propagations'][0]],
            reliabilities=[quantiles[3]],
            previous_virtual_age=parameters['previousVirtualAge'],
        )['times'][0]

        for i in range(1, n_cm):
            ql[i] = (
                cum_cm[i - 1]
                + qwgrp(
                    1,
                    parameters['a'],
                    parameters['b'],
                    parameters['q'],
                    [parameters['propagations'][i]],
                    reliabilities=[quantiles[0]],
                    previous_virtual_age=conditional_means['virtualAges'][
                        i - 1
                    ],
                )['times'][0]
            )
            qu[i] = (
                cum_cm[i - 1]
                + qwgrp(
                    1,
                    parameters['a'],
                    parameters['b'],
                    parameters['q'],
                    [parameters['propagations'][i]],
                    reliabilities=[quantiles[1]],
                    previous_virtual_age=conditional_means['virtualAges'][
                        i - 1
                    ],
                )['times'][0]
            )
            qc[i] = (
                cum_cm[i - 1]
                + qwgrp(
                    1,
                    parameters['a'],
                    parameters['b'],
                    parameters['q'],
                    [parameters['propagations'][i]],
                    reliabilities=[quantiles[2]],
                    previous_virtual_age=conditional_means['virtualAges'][
                        i - 1
                    ],
                )['times'][0]
            )
            qn[i] = (
                cum_cm[i - 1]
                + qwgrp(
                    1,
                    parameters['a'],
                    parameters['b'],
                    parameters['q'],
                    [parameters['propagations'][i]],
                    reliabilities=[quantiles[3]],
                    previous_virtual_age=conditional_means['virtualAges'][
                        i - 1
                    ],
                )['times'][0]
            )
            cum_cm[i] = cum_cm[i - 1] + conditional_means['mean'][i]

    # n_x = len(x) if x else 0
    # n_x = len(x) if not x.empty else 0
    if len(x) > 0:
        n_x = len(x)
    else:
        n_x = 0

    bs_nrow = 0
    mean_cum_bs = None
    sql = None
    squ = None

    if n_x > 0:
        cum_x = np.zeros(n_x)
        cum_x[0] = x[0]
        for i in range(1, n_x):
            cum_x[i] = cum_x[i - 1] + x[i]

    bs_ncol = 0
    cum_bs = None
    events_in_the_future_tense = bootstrap_sample['events_in_the_future_tense']
    bootstrap_sample = bootstrap_sample['sample_matrix']
    if bootstrap_sample is not None and len(bootstrap_sample) > 0:
        bs_nrow = bootstrap_sample.shape[0]
        bs_ncol = bootstrap_sample.shape[1]
        cum_bs = np.zeros((bs_nrow, bs_ncol))
        for i in range(bs_nrow):
            cum_bs[i] = np.cumsum(bootstrap_sample[i])
        mean_cum_bs = np.mean(cum_bs, axis=0)
        squ = np.percentile(cum_bs, quantiles[0] * 100, axis=0)
        sql = np.percentile(cum_bs, quantiles[1] * 100, axis=0)
        # aqui mechi 13/07
        
        best_quantile = None
        min_mse = float('inf')

        # Iterar sobre cada quantil e calcular o MSE
        
        real_serie = accumulate_values(x)
        for quantile in quantiles:
            tmp = list(np.percentile(cum_bs, quantile * 100, axis=0))
            mse = _root_mean_squared_error(real_serie, tmp[:len(x)])
            
            if mse < min_mse:
                min_mse = mse
                best_quantile = quantile
        
        nql  = np.percentile(cum_bs, best_quantile * 100, axis=0)
        
        
        # Initialize lists to store the best series and their respective MSEs
        best_series = []
        mse_series = []

        # For each series generated from the bootstrap sample
        for i in range(bootstrap_sample.shape[0]):
            rando_serie = accumulate_values(bootstrap_sample[i, :])
            
            # Calculate the MSE by comparing the generated series with the real series
            mse = _root_mean_squared_error(real_serie, rando_serie[:len(real_serie)])
            
            # Add the series and MSE to the best series list if there's still space
            if len(mse_series) < top_n_series:
                mse_series.append(mse)
                best_series.append(rando_serie)
            else:
                # Check if the new MSE is smaller than the largest MSE in the list (worst of the best series)
                max_mse_index = mse_series.index(max(mse_series))
                if mse < mse_series[max_mse_index]:
                    # Replace the series with the largest MSE with the new series
                    mse_series[max_mse_index] = mse
                    best_series[max_mse_index] = rando_serie

        # Calculate the minimum length among all the best series to avoid index out of range issues
        min_length = min([len(serie) for serie in best_series])

        # Calculate the point-by-point average among the best series
        list_best_prediction = [sum(best_series[j][i] for j in range(top_n_series)) / top_n_series for i in range(min_length)]

        


    res = {
        'eventsInTheFutureTense': events_in_the_future_tense,
        'cumTimes': cum_x.tolist() if cum_x is not None else None,
        'cumQuantile': cum_q.tolist() if cum_q is not None else None,
        'cumCondMeans': cum_cm.tolist() if cum_cm is not None else None,
        'cumBootstraps': cum_bs.tolist() if cum_bs is not None else None,
        'meanBootstraps': mean_cum_bs.tolist()
        if mean_cum_bs is not None
        else None,
        'sampleLowerBound': sql.tolist() if sql is not None else None,
        'sampleUpperBound': squ.tolist() if squ is not None else None,
        'sampleCentralBound': None,
        'lowerBound': ql.tolist() if ql is not None else None,
        'upperBound': qu.tolist() if qu is not None else None,
        'centralBound': qc.tolist() if qc is not None else None,
        'newQuantile': nql.tolist() if nql is not None else None,
        'inicialTime': real_serie[0],
        'best_prediction' : list_best_prediction,
        'best_quantile' : best_quantile
    }

    return res


def add_time_diffs(data, time_unit):
    if time_unit not in [
        'days',
        'seconds',
        'microseconds',
        'milliseconds',
        'minutes',
        'hours',
        'weeks',
    ]:
        raise ValueError(
            "Invalid time_unit. Expected 'days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', or 'weeks'."
        )

    if time_unit == 'days':
        divider = timedelta(days=1)
    elif time_unit == 'seconds':
        divider = timedelta(seconds=1)
    elif time_unit == 'microseconds':
        divider = timedelta(microseconds=1)
    elif time_unit == 'milliseconds':
        divider = timedelta(milliseconds=1)
    elif time_unit == 'minutes':
        divider = timedelta(minutes=1)
    elif time_unit == 'hours':
        divider = timedelta(hours=1)
    elif time_unit == 'weeks':
        divider = timedelta(weeks=1)
    # Parse dates
    dates = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
    data_sorted = data.sort_values(by='date')
    dates_sorted = dates.sort_values()

    # Calculate time differences in hours (decimal)
    n_dates = len(dates_sorted)
    TBEs = dates_sorted.diff() / divider   # Convert timedelta to hours (float)
    TBEs.iloc[0] = pd.NA

    # Add TBEs to data
    data_sorted['TBE'] = TBEs
    data_sorted

    return data_sorted.dropna()


def _compute_forecasting_table(
    forecasting, initial_time=10, failure_times=None
):
    initial_time = forecasting['inicialTime']
    n = len(forecasting['cumTimes'])
    m = len(forecasting['meanBootstraps']) - n

    intervention = list(range(1, n + m + 1))
    lower = [
        round(value + initial_time, 2)
        for value in forecasting['sampleLowerBound']
    ]

    sample_central_bound = forecasting.get('sampleCentralBound', [])
    if sample_central_bound is None:
        sample_central_bound = []

    pib = [round(value + initial_time, 2) for value in sample_central_bound]

    mean = [
        round(value + initial_time, 2)
        for value in forecasting['meanBootstraps']
    ]
    upper = [
        round(value + initial_time, 2)
        for value in forecasting['sampleUpperBound']
    ]
    new = [
        round(value + initial_time, 2) for value in forecasting['newQuantile']
    ]
    list_best_prediction = [
        round(value + initial_time, 2) for value in forecasting['best_prediction']
    ] 

    ret = pd.DataFrame(
        {
            'Intervention': intervention,
            'Quantile_2.5': lower,
            'Mean': mean,
            'Quantile_97.5': upper ,
            'newQuantile': new,
            'best_prediction' : list_best_prediction
        }
    )

    return {'dataframe': ret, 'n_forecasts_events': forecasting['eventsInTheFutureTense'], 'best_quantile': forecasting['best_quantile'] }


def _summarize_ics_and_parameters_table(mle_objs, x, nDecs=2):

    df = pd.DataFrame()

    m = len(mle_objs)
    for i in range(m):
        mle_i = mle_objs[i]
        NM_i = mle_i['parameters']['formalism']
        IC_i = ic_wgrp(mle_i, x)

        row_i = {
            'Formalism': NM_i,
            'AIC': IC_i['AIC'],
            'AICc': IC_i['AICc'],
            'BIC': IC_i['BIC'],
            'LL': IC_i['LL'],
            'alpha': mle_i['a'],
            'beta': mle_i['b'],
            'q': mle_i['q'],
            'y_prev': None,
            'y_corr': None,
            
        }

        if NM_i == FORMALISM['KIJIMA_I']:
            row_i['y_prev'] = 1
            row_i['y_corr'] = 1
        elif NM_i == FORMALISM['KIJIMA_II']:
            row_i['y_prev'] = 0
            row_i['y_corr'] = 0
        elif NM_i == FORMALISM['INTERVENTION_TYPE']:
            row_i['y_prev'] = mle_i['optimum'][1]
            row_i['y_corr'] = mle_i['optimum'][2]
        df = pd.concat([df, pd.DataFrame([row_i])], ignore_index=True)

    bestIndex = df['BIC'].idxmin()
    df2 = df.copy()
    df2.iloc[:, 1:] = df2.iloc[:, 1:].round(nDecs)
    df3 = df2.loc[bestIndex].to_frame().T

    return {'df1': df2, 'df2': df3}
