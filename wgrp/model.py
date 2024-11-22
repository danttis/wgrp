import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wgrp.compute import _cumulative_forecast_times, bootstrap_sample, add_time_diffs, _summarize_ics_and_parameters_table, _compute_forecasting_table, accumulate_values
from wgrp.base_functions import Get, Parameters
from wgrp.getcomputer import _getMLE_objs
from wgrp.moments import _sample_conditional_moments


def _fit(data, time_unit='days', cumulative=False, random_state=0, optimizer="ps"):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()  

    if pd.api.types.is_numeric_dtype(data) or all(isinstance(x, (int, float)) for x in data):
        type = 'numeric'
    else:
        try:
            #data = pd.to_datetime(data)
            type = 'date'
        except (ValueError, TypeError):
            raise ValueError("Invalid type. Expected 'date' or 'numeric'.")
    
    np.random.seed(random_state)

    if type == 'date':
        newData_i = add_time_diffs(
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
      
    mle_objs = _getMLE_objs(
        timesBetweenInterventions=list(TBEs),
        interventionsTypes=event_types,
        b=1,
        random_state = random_state,
        optimizer=optimizer
    )

    return mle_objs, TBEs

PROPAGATION = Parameters().PROPAGATION
get_parameters = Get().get_parameters
get_optim = Get().get_optimum


def _pred(n_forecasts, mle_objs, TBEs, n_steps_ahead = 0, random_series=10000, top_n_series=3):
    df = _summarize_ics_and_parameters_table(mle_objs, TBEs)[
        'df1'
    ]
    
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
    m = n_forecasts
    

    pmPropagations = np.concatenate(
        (optimum['propagations'], np.repeat(PROPAGATION['KijimaII'], m))
    )
    parameters = get_parameters(
        nSamples=random_series,
        nInterventions=(n + m),
        a=optimum['a'],
        b=optimum['b'],
        q=optimum['q'],
        propagations=pmPropagations,
        cumulativeFailureCount=cF,
        timesPredictFailures=n_steps_ahead,
        nIntervetionsReal=n,
    )
   
    bSample = bootstrap_sample(parameters)
    theoreticalMoments = _sample_conditional_moments(parameters)

    forecasting = _cumulative_forecast_times(
        x=TBEs,
        bootstrap_sample=bSample,
        conditional_means=theoreticalMoments,
        parameters=parameters,
        probability_of_failure=n_forecasts,
        top_n_series=top_n_series
    ) 

    forecasting_final = _compute_forecasting_table(forecasting, initial_time=10)

    return forecasting_final, optimum, df, parameters


class wgrp_model:
    """The `wgrp_model` class is the main function of the package and it controls all other functions.
    Although all other functions can be used separately, this class provides two main functions:

    - `fit`: Works similarly to many machine learning packages, fitting WGRP models to the times between 
    events (TBEs) data and returning a DataFrame with the parameters of a 
    number of WGRP formalisms (i.e. Renew Processes - RP, Non-Homogeneous Poisson Processes - NHPP, Kijima I, 
    Kijima II, and Intervention type-based models). Further, a list with the TBEs is returned.
    - `predict`: Also works similarly to machine learning packages, receiving the number of events for 
    which the times until occurrence must be forecasted (i.e. out-of-sample predictions) and returning a 
    DataFrame with four columns: the index of each event, the 2.5% quantile (i.e. the lower bound of the
      95% confidence interval), the 97.5% quantile (i.e. the upper bound of the 95% confidence interval), 
      and the mean value of the times to occur the events under study.

    - Objects:
        The class provides some objects used in modeling that are also directly accessible:
            - `TBEs_`: contains the values ​​collected from the times between events (TBEs).
            - `optimum_`: stores the parameters of the best selected formalism, generally used for the options in the `predict` function.
            - `mle_objs_`: similar to `optimum_`, but includes the parameters of all formalisms used (RP, Non-Homogeneous Poisson Processes - NHPP, Kijima I, Kijima II and models based on intervention types).
            - `df_`: a DataFrame that presents the series data, the best modeling for the configurations, and the performance measures of each formalism, such as AIC, AICc, BIC and Log-Likelihood (LL).
    """

    def __init__(self):
        self.TBEs_ = None    # Attributes to store the fitting results
        self.name = None
        self.mle_objs_ = None
        self.optimum_ = None
        self.df_ = None
        self.predictions = None
        self.time_unit = None
        self.parameters = None
        self.n_steps_ahead = None

    def fit(self, data, time_unit='days', cumulative=False, random_state=0, optimizer="ps"):
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
            time_unit (str):
                Time unit for analyzing intervals between interventions. It can be 'weeks', 'days', 'hours', 
                'minutes', 'seconds', 'microseconds', 'milliseconds'. Default is 'days'.
            cumulative (bool):
                Indicates if the provided numeric times are cumulative. Default is `False`. there are two 
                options for `data": TBEs if `cumulative = False` (e.g. [2, 4, 3, 5]), or  
                `cumulative = True` (e.g. [2, 6, 9, 14]). 
            optimizer (str): 
                Selects the type of optimizer to use, either "ps" for particle swarm (default) or "sa" for simulated annealing.



        Examples:
            >>> TBEs = [0.2, 1, 5, 7, 89, 21, 12]
            >>> model = wgrp_model()
            >>> model.fit(TBEs, time_unit='minutes')
        """
        # Calls the fit_f method of Fit_grp to fit the model
        self.time_unit = time_unit
        self.mle_objs_, self.TBEs_ = _fit(data, time_unit, cumulative, random_state, optimizer)

    def predict(self, n_forecasts=1, n_steps_ahead=0, random_series=10000, top_n_series=3):
            """
            Makes future (out-of-sample) forecasts based on the desired number of steps ahead.

            Attributes:
                self.optimum_: Stores the optimum value calculated during the prediction process. It is 
                updated with each call of the predict function.
                self.df_: Stores the DataFrame used in the prediction calculations. It is updated with new 
                predictions each time the predict function is called.

            Parameters:
                n_forecasts (int): Number of future events to be calculated.
                n_steps_ahead (int, optional): Number of events to be considered in the future. 
                Default is 0.
                random_series (int, optional): Specifies the number of random series to be generated for the 
                prediction process. The default is 10,000. This method generates random series and averages 
                them to return a more robust prediction. Increasing this number may lead to slower performance 
                but potentially more accurate results.
                top_n_series (int, optional): Specifies the number of best series (with the lowest RMSE) to be 
                selected for averaging. The default is 3. The final prediction will be the average of these 
                top series.

            Returns:
                Array: Returns the n-step-ahead forecast estimated from the best series defined in `top_n_series`.

            Examples:
                >>> TBEs = [0.2, 1, 5, 7, 89, 21, 12]
                >>> model = wgrp_model()
                >>> model.fit(TBEs, time_unit='minutes')
                >>> predictions = model.predict(3)
                alpha = 1.1910044773056132
                beta = 0.41122725565015567
                q = 1
            """

            self.predictions, self.optimum_, self.df_, self.parameters = _pred(
                n_forecasts, list(self.mle_objs_), list(self.TBEs_), n_steps_ahead, random_series, top_n_series
            )
            return self.predictions['dataframe']['best_prediction'].iloc[len(self.TBEs_)-1:].values
    
    def plot(self, n_random_series=10):
        """
        Plots a comparison of real series, bootstrapped series, and predicted quantiles with optional random series.

        Parameters:
            n_random_series : int, optional
                The number of random series to be generated from the bootstrap samples. Default is 10.

        Description:
            This function generates a plot that displays:
            - Randomly generated series based on bootstrapped samples.
            - The observed real series.
            - Predicted quantiles (upper and lower) and the mean of the WGRP predictions.
            - The 'best quantile' and the 'best prediction'.
            - The end of the training data marked with a vertical dashed green line.

        The plot visually compares the actual data with the predictions and quantiles, giving insights into how well 
        the model fits the observed series.
        
        Returns:
            None: The function displays a matplotlib plot.
        """

        # Set the number of samples for the bootstrap
        self.parameters['nSamples'] = n_random_series

        # Generate random series using bootstrap sampling
        random_series = bootstrap_sample(self.parameters)['sample_matrix']

        # Accumulate real series values
        real_serie = accumulate_values(self.TBEs_)

        best_quantile = self.predictions['best_quantile']
        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        # Plot each random series in gray with a thin line
        for i in range(random_series.shape[0]):
            ax.plot(accumulate_values(random_series[i, :]), label=None, color='gray', linewidth=0.5)

        # Plot the predicted upper quantile (97.5%)
        ax.plot(self.predictions['dataframe']['Quantile_97.5'], label='Theoretical Quantiles', color='black')

        # Plot the predicted mean of the WGRP model
        ax.plot(self.predictions['dataframe']['Mean'], label='WGRP Mean', color='red')

        # Plot the predicted lower quantile (2.5%)
        ax.plot(self.predictions['dataframe']['Quantile_2.5'], label=None, color='black')

        # Plot the 'best quantile' in orange
        ax.plot(self.predictions['dataframe']['newQuantile'], label=f'Best Quantile {best_quantile}', color='orange')

        # Plot the 'best prediction' in green
        ax.plot(self.predictions['dataframe']['best_prediction'], label='Best Prediction', color='green')

        # Plot the real observed series in blue
        ax.plot(real_serie, label='Observed Series', color='blue')

        # Mark the end of the training data with a vertical dashed line
        ax.axvline(x=len(real_serie) - 1, color='green', linestyle='--', label='End of Train Data')

        # Display the legend
        ax.legend()

        # Set axis labels
        ax.set_xlabel('Occurrences')
        ax.set_ylabel(f'Time in {self.time_unit}')

        # Show the plot
        plt.show()



