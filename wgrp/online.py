import numpy as np
from scipy import optimize as _optimize

from wgrp.base_functions import Get, Parameters
from wgrp.mle_wgrp import MleWgrp
from wgrp.model import _pred
from wgrp.virtual_ages import virtual_age
from wgrp.wgrp_functions import dwgrp, ic_wgrp

FORMALISM = Parameters().FORMALISM

_ONLINE_FORMALISMS = {
    FORMALISM['RP']: {'q': 0.0, 'propagations': None, 'free': ('b',)},
    FORMALISM['NHPP']: {'q': 1.0, 'propagations': None, 'free': ('b',)},
    FORMALISM['KIJIMA_I']: {'q': None, 'propagations': 1.0, 'free': ('b', 'q')},
    FORMALISM['KIJIMA_II']: {'q': None, 'propagations': 0.0, 'free': ('b', 'q')},
}

# same order of the standard package fits (mle_objs indices match)
_AUTO_ORDER = [
    FORMALISM['RP'],
    FORMALISM['NHPP'],
    FORMALISM['KIJIMA_I'],
    FORMALISM['KIJIMA_II'],
]


class _StreamingTracker:
    """
    Streaming state of a single WGRP formalism: the O(1) statistics (running sum
    `S`, virtual age Markov chain, prequential log-likelihood), the closed-form
    scale parameter `a`, and the warm-started local search that tracks `(b, q)`.
    """

    def __init__(self, formalism, random_state=0, maxiter=100):
        self.formalism = formalism
        self.spec = _ONLINE_FORMALISMS[formalism]
        self.random_state = random_state
        self.maxiter = maxiter
        self.n_evals = 0

        self.b = 1.0
        self.q = 0.5 if self.spec['q'] is None else self.spec['q']
        self.propagations = self.spec['propagations']
        self._reset_state()

    def _reset_state(self):
        self.a = None
        self.n = 0
        self.virtual_age = 0.0
        self.S = 0.0
        self.loglik = 0.0
        self.v_prevs = []

    def propagation_for(self, index):
        """
        Propagation (weight between Kijima I and II) of the intervention `index`,
        following the same conventions of the standard package: `None`
        propagations lead to -1 (as in `get_sample_virtual_ages`), scalar
        propagations are constant, and arrays are indexed by intervention
        (observations beyond the array reuse its last value).
        """
        propagations = self.propagations
        if propagations is None:
            return -1
        if np.isscalar(propagations):
            return float(propagations)
        arr = np.asarray(propagations, dtype=float)
        return arr[index] if index < len(arr) else arr[-1]

    def update_state(self, x):
        """
        Ingests one new TBE into the exact streaming state in O(1) -- the
        log-likelihood term of `x` under the current parameters, the running sum
        `S`, the closed-form scale parameter `a`, and the virtual age Markov
        chain. Identical operations (and order) of `_get_virtual_ages_and_a` and
        `lwgrp` on the same data prefix.
        """
        v_prev = self.virtual_age

        # Log-likelihood increment (scored under the parameters estimated so far)
        if self.a is not None:
            self.loglik += dwgrp(x, self.a, self.b, v_prev)

        # Running sum S
        if self.b != 0:
            current_value = x + v_prev
            previous_virtual_age = v_prev
            if current_value < 0:
                current_value = 0
            if previous_virtual_age < 0:
                previous_virtual_age = 0
            self.S += np.power(current_value, self.b) - np.power(
                previous_virtual_age, self.b
            )

        self.n += 1

        # Closed-form MLE of a given (b, q, propagations) -- re-estimated from
        # the running sum at every step (exact prefix value)
        if self.b != 0:
            self.a = np.power(self.S / self.n, 1 / self.b)

        # Virtual age Markov update
        self.virtual_age = virtual_age(
            self.propagation_for(self.n - 1), self.q, v_prev, x
        )['virtualAge']

        self.v_prevs.append(v_prev)

    def replay(self, tbes):
        """Rebuilds the exact streaming state by replaying the history (O(n))
        under the current parameters."""
        self._reset_state()
        for x in tbes:
            self.update_state(x)

    def objective(self, tbes):
        """
        Negative log-likelihood of the history, as a function of the free
        parameters of the formalism (`[b]` or `[b, q]`), via the same
        `MleWgrp.objective_function` of the standard package.
        """
        p_parameters = Get().get_parameters(
            b=self.b, q=self.q, formalism=self.formalism
        )
        mle = MleWgrp(
            x=list(tbes),
            p_parameters=p_parameters,
            random_state=self.random_state,
            optimizer='nm',
        )
        return mle.objective_function

    def local_optimize(self, tbes):
        """
        One warm-started local search (bounded Nelder-Mead) around the current
        `(b, q)`. Accepted only if it improves the log-likelihood, so the
        streaming estimates never degrade.
        """
        free_q = 'q' in self.spec['free']
        x0 = [self.b, self.q] if free_q else [self.b]
        bounds = [(Parameters().bBounds['min'], Parameters().bBounds['max'])]
        if free_q:
            bounds.append(
                (Parameters().qBounds['min'], Parameters().qBounds['max'])
            )

        objective = self.objective(tbes)
        f0 = objective(x0)
        self.n_evals += 1

        result = _optimize.minimize(
            objective,
            x0=x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': self.maxiter, 'xatol': 1e-6, 'fatol': 1e-8},
        )
        self.n_evals += result.nfev

        if result.fun < f0 and np.isfinite(result.fun):
            self.b = float(result.x[0])
            if free_q:
                self.q = float(result.x[1])
            self.replay(tbes)
            return True
        return False

    def loglikelihood(self, tbes):
        """
        Batch-equivalent log-likelihood of the history under the current
        estimates: the same value `lwgrp` returns. Unlike the prequential
        `loglik`, every observation is scored under the current parameters.
        """
        if self.a is None or self.n == 0:
            return None
        l = 0.0
        for x, v_prev in zip(tbes, self.v_prevs):
            l += dwgrp(x, self.a, self.b, v_prev)
        return l

    def mle_obj(self, tbes):
        """
        Synthesizes, from the streaming state, the same model object the
        standard package returns from its fits (keys 'a', 'b', 'q',
        'propagations', 'optimum', 'parameters', 'optimum_value').
        """
        if self.a is None or self.n == 0:
            raise ValueError('No data ingested yet.')

        if self.spec['propagations'] is None and self.propagations is None:
            propagations = None
        elif self.propagations is not None and not np.isscalar(
            self.propagations
        ):
            propagations = np.asarray(self.propagations, dtype=float)
        else:
            propagations = np.full(
                self.n,
                self.spec['propagations']
                if self.propagations is None
                else float(self.propagations),
            )

        optimum = [self.b, self.q] if 'q' in self.spec['free'] else [self.b, self.spec['q']]

        return {
            'a': self.a,
            'b': self.b,
            'q': self.q,
            'propagations': propagations,
            'virtualAges': None,
            'optimum': optimum,
            'parameters': Get().get_parameters(
                b=self.b, q=self.q, formalism=self.formalism
            ),
            'optimum_value': self.loglikelihood(tbes),
        }


class wgrp_online_model:
    """
    True online (streaming) version of the `wgrp_model` class.

    Just like the online linear regression update -- in which running statistics
    are updated in O(1) per new observation -- this class ingests the times
    between events (TBEs) one at a time and never refits from scratch over the
    whole history:

    - Virtual age (`Kijima` mixed model): a Markov chain,
      `v_i = p_i * (v_{i-1} + q * x_i) + (1 - p_i) * q * (v_{i-1} + x_i)`,
      which only depends on the previous virtual age -> updated in O(1).
    - Scale parameter `a`: given `(b, q, propagations)`, its MLE has closed form,
      `a = (S_n / n) ** (1 / b)`, where
      `S_n = sum_i [ (x_i + v_{i-1}) ** b - v_{i-1} ** b ]`
      is a running sum -> updated in O(1) and exactly equal to the value
      `_get_virtual_ages_and_a` computes on the same prefix of data.
    - Log-likelihood: a sum of per-observation terms, each one depending only on
      the previous virtual age -> updated in O(1).
    - Shape parameter `b` and rejuvenation parameter `q`: no closed-form
      estimator exists, so they are tracked by *warm-started local optimization*
      (bounded Nelder-Mead starting from the current `(b*, q*)` and accepting
      only improvements), the standard recursive-estimation scheme: each new
      event only triggers a cheap local search around the previous solution,
      never a global refit.

    By default (`formalism='auto'`) the four supported formalisms -- RP, NHPP,
    Kijima I and Kijima II -- are tracked in parallel and the best one is
    selected by the BIC at every event, mirroring the model selection of the
    standard `wgrp_model`. One can also track a single formalism by passing its
    name.

    Objects:
        - `TBEs_`, `event_types_`, `n_`: history of times between events,
          intervention types and number of TBEs ingested so far.
        - `formalism_`: formalisms tracked online ('auto' or a single name).
        - `a_`, `b_`, `q_`, `propagations_`, `virtual_age_`, `S_`: streaming
          estimates and state of the active (BIC-best) formalism.
        - `loglik_`: streaming (prequential) log-likelihood of the active
          formalism: each observation is scored under the parameters in use at
          the moment it arrives.
        - `optimum_`: model object of the active (BIC-best) formalism.
        - `n_evals_`: cumulative number of log-likelihood evaluations spent by
          the local searches (a measure of the online computational cost).

    Examples:
        >>> TBEs = [0.2, 1, 5]
        >>> model = wgrp_online_model()
        >>> _ = model.update(0.2)
        >>> _ = model.update(1)
        >>> _ = model.update(5)
        >>> model.n_
        3
    """

    def __init__(
        self,
        formalism='auto',
        random_state=0,
        optimize_every=1,
        optimize_from=3,
        maxiter=100,
    ):
        """
        Parameters:
            formalism (str): 'auto' (default) tracks RP, NHPP, Kijima I and
                Kijima II in parallel and selects the best by the BIC at every
                event; alternatively, the name of a single formalism to track.
            random_state (int): seed kept for compatibility with the standard
                package (the local searches are deterministic).
            optimize_every (int): how often the warm-started local searches run:
                every `optimize_every`-th ingested TBE (1 = every event). Use a
                larger value (or `0`) to only update the O(1) state.
            optimize_from (int): minimum number of TBEs before the local
                searches start. Default is 3.
            maxiter (int): maximum iterations of each warm-started Nelder-Mead
                search. Default is 100.
        """
        if formalism == 'auto':
            names = list(_AUTO_ORDER)
        elif formalism in _ONLINE_FORMALISMS:
            names = [formalism]
        else:
            raise ValueError(
                f"Invalid formalism '{formalism}'. Expected 'auto' or one of "
                f"{sorted(_ONLINE_FORMALISMS)}."
            )
        self.formalism_ = formalism
        self.random_state = random_state
        self.optimize_every = optimize_every
        self.optimize_from = optimize_from
        self.maxiter = maxiter
        self.time_unit = 'days'

        self._trackers = {
            name: _StreamingTracker(
                name, random_state=random_state, maxiter=maxiter
            )
            for name in names
        }
        self._active = names[0]

        # Shared history
        self.TBEs_ = []
        self.event_types_ = []
        self.n_ = 0

        # Results of the last predict call
        self.predictions = None
        self.df_ = None
        self.parameters = None
        self.n_steps_ahead = None
        self.optimum_ = None

    # -- active (BIC-best) tracker -------------------------------------

    def _active_tracker(self):
        return self._trackers[self._active]

    def _select_bic(self):
        """Selects the active formalism as the one with minimum BIC (the same
        criterion of the standard `wgrp_model`)."""
        best_name, best_bic = None, np.inf
        for name, tracker in self._trackers.items():
            obj = tracker.mle_obj(self.TBEs_)
            bic = ic_wgrp(obj, self.TBEs_)['BIC']
            if bic < best_bic:
                best_name, best_bic = name, bic
        if best_name is not None:
            self._active = best_name
        self.optimum_ = self._active_tracker().mle_obj(self.TBEs_)

    # -- streaming attributes of the active formalism ------------------

    @property
    def a_(self):
        return self._active_tracker().a

    @property
    def b_(self):
        return self._active_tracker().b

    @property
    def q_(self):
        return self._active_tracker().q

    @property
    def propagations_(self):
        return self._active_tracker().propagations

    @property
    def virtual_age_(self):
        return self._active_tracker().virtual_age

    @property
    def S_(self):
        return self._active_tracker().S

    @property
    def loglik_(self):
        return self._active_tracker().loglik

    @property
    def n_evals_(self):
        return sum(t.n_evals for t in self._trackers.values())

    def loglikelihood(self):
        """
        Batch-equivalent log-likelihood of the ingested history under the
        current estimates of the active formalism: the same value `lwgrp`
        returns.

        Returns:
            float or None: the log-likelihood (None if there is no estimate `a`
            yet).
        """
        return self._active_tracker().loglikelihood(self.TBEs_)

    # -- streaming ------------------------------------------------------

    def update(self, x, event_type='Corrective', optimize=None):
        """
        Ingests one new time between events (TBE):

        1. updates the exact streaming state of every tracked formalism in O(1)
           -- the log-likelihood term of `x` under the current parameters, the
           running sum `S` and, in closed form, the scale parameter `a`, and the
           virtual age Markov chain;
        2. (by default) re-estimates `(b*, q*)` of every tracked formalism with
           a warm-started local search around the current values, never
           refitting from scratch;
        3. with `formalism='auto'`, re-selects the active formalism by the BIC.

        Parameters:
            x (float): the new time between events.
            event_type (str): 'Corrective' (default) or 'Preventive'.
            optimize (bool or None): whether to run the local searches for this
                event. None (default) follows `optimize_every`/`optimize_from`.

        Returns:
            dict: the current streaming state and estimates of the active
            formalism {'a', 'b', 'q', 'n', 'virtual_age', 'S', 'loglik'}.

        Examples:
            >>> model = wgrp_online_model()
            >>> state = model.update(3.2)
            >>> state['n']
            1
        """
        self.TBEs_.append(x)
        self.event_types_.append(event_type)
        self.n_ += 1

        for tracker in self._trackers.values():
            tracker.update_state(x)

        if optimize is None:
            optimize = (
                self.optimize_every > 0
                and self.n_ >= self.optimize_from
                and (self.n_ - self.optimize_from) % self.optimize_every == 0
            )
        if optimize:
            for tracker in self._trackers.values():
                tracker.local_optimize(self.TBEs_)

        if len(self._trackers) > 1:
            self._select_bic()
        else:
            self.optimum_ = self._active_tracker().mle_obj(self.TBEs_)

        tracker = self._active_tracker()
        return {
            'a': tracker.a,
            'b': tracker.b,
            'q': tracker.q,
            'n': self.n_,
            'virtual_age': tracker.virtual_age,
            'S': tracker.S,
            'loglik': tracker.loglik,
        }

    def set_parameters(
        self, formalism=None, b=None, q=None, propagations='unset'
    ):
        """
        Sets the streaming parameter estimates of one tracked formalism and
        rebuilds its exact streaming state by replaying the ingested history
        (O(n)). The scale parameter `a` is always recomputed in closed form.

        Parameters:
            formalism (str): which tracked formalism to set (required when
                tracking more than one, i.e. `formalism='auto'`).
            b (float): shape parameter.
            q (float): rejuvenation parameter.
            propagations: weights between Kijima I and II (None, scalar or
                list). The string 'unset' (default) keeps the current ones.
        """
        if formalism is None:
            if len(self._trackers) > 1:
                raise ValueError(
                    'Multiple formalisms are being tracked: pass the '
                    "'formalism' argument."
                )
            formalism = self._active
        if formalism not in self._trackers:
            raise ValueError(
                f"Formalism '{formalism}' is not being tracked "
                f"({sorted(self._trackers)})."
            )

        tracker = self._trackers[formalism]
        if b is not None:
            tracker.b = b
        if q is not None:
            tracker.q = q
        if propagations != 'unset':
            tracker.propagations = propagations
        tracker.replay(self.TBEs_)

        if len(self._trackers) > 1:
            self._select_bic()

    def fit(self, data, cumulative=False):
        """
        Ingests a full dataset as a stream (one `update` -- with the
        warm-started local searches -- per TBE). The estimates evolve event by
        event and are never refit from scratch.

        Parameters:
            data (list of float): times between events (if `cumulative = True`,
                cumulative times, converted to TBEs as in the standard `fit`).
            cumulative (bool): whether the numeric times are cumulative.

        Returns:
            dict: the final streaming state and estimates of the active
            formalism.

        Examples:
            >>> TBEs = [0.2, 1, 5, 7, 89, 21, 12]
            >>> model = wgrp_online_model(optimize_from=3, maxiter=20)
            >>> state = model.fit(TBEs)
            >>> state['n']
            7
        """
        if cumulative:
            data = [data[i + 1] - data[i] for i in range(len(data) - 1)]
        state = None
        for x in data:
            state = self.update(x)
        return state

    # -- standard-package interface ------------------------------------

    def mle_obj(self):
        """
        Synthesizes, from the streaming state of the active (BIC-best)
        formalism, the same model object the standard package returns from its
        fits, which allows using the standard information criteria and
        prediction machinery online.

        Returns:
            dict: a WGRP model object for the current streaming estimates.
        """
        if self.n_ == 0:
            raise ValueError(
                'No data ingested yet. Call update(x) or fit(data) first.'
            )
        return self._active_tracker().mle_obj(self.TBEs_)

    def mle_objs(self):
        """
        Same as `mle_obj`, but for every tracked formalism (in the same order
        the standard package fits them when `formalism='auto'`).

        Returns:
            list of dict: WGRP model objects for the current streaming
            estimates.
        """
        if self.n_ == 0:
            raise ValueError(
                'No data ingested yet. Call update(x) or fit(data) first.'
            )
        return [t.mle_obj(self.TBEs_) for t in self._trackers.values()]

    def predict(self, n_forecasts=1, n_steps_ahead=0, random_series=10000, top_n_series=3):
        """
        Same `predict` function of the standard `wgrp_model` (see its
        documentation), computed from the current streaming estimates.

        Returns:
            Array: the n-step-ahead forecast estimated from the best series
            defined in `top_n_series`.
        """
        if self.n_ == 0:
            raise ValueError(
                'No data ingested yet. Call update(x) or fit(data) first.'
            )

        mle_objs = self.mle_objs()

        self.predictions, self.optimum_, self.df_, self.parameters = _pred(
            n_forecasts,
            mle_objs,
            list(self.TBEs_),
            n_steps_ahead,
            random_series,
            top_n_series,
        )
        return (
            self.predictions['dataframe']['best_prediction']
            .iloc[len(self.TBEs_) - 1:]
            .values
        )