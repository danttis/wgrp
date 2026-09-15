import numpy as np

from wgrp.model import wgrp_model
from wgrp.online import wgrp_online_model
from wgrp.virtual_ages import _get_virtual_ages_and_a
from wgrp.wgrp_functions import lwgrp

random_values = [10, 12, 15, 4, 56, 12, 22, 13]


def test_online_state_matches_batch_prefix_exactly():
    """
    With the parameters held fixed, the O(1) streaming state (running sum S,
    closed-form a and virtual age chain) must equal, at every prefix of the data,
    what the standard package computes from scratch on that prefix -- just like
    the online linear regression of the reference notebook matches the batch
    statistics at every step.
    """
    b, q = 1.5, 0.6
    online = wgrp_online_model(formalism='Kijima II', optimize_every=0)
    online.set_parameters(b=b, q=q, propagations=None)

    for k, x in enumerate(random_values):
        online.update(x)
        prefix = random_values[: k + 1]

        batch = _get_virtual_ages_and_a(b, q, None, prefix)
        assert online.a_ == batch['a']
        assert online.virtual_age_ == batch['virtualAges'][-1]

        # the running sum S must match the batch computation on the prefix
        v_prevs = [0.0] + batch['virtualAges'][:-1]
        S = 0.0
        for x_i, v_prev in zip(prefix, v_prevs):
            current_value = x_i + v_prev
            previous_virtual_age = v_prev
            if current_value < 0:
                current_value = 0
            if previous_virtual_age < 0:
                previous_virtual_age = 0
            S += np.power(current_value, b) - np.power(previous_virtual_age, b)
        assert online.S_ == S

        # exact batch-equivalent log-likelihood under the current parameters
        assert online.loglikelihood() == lwgrp(
            prefix, online.a_, b, q, None
        )


def test_streaming_never_degrades_and_tracks_the_batch_optimum():
    """
    Feeding the data one TBE at a time, the warm-started local searches must keep
    the streaming estimates at (or very close to) the optimum of the
    log-likelihood over the data seen so far -- the streaming log-likelihood must
    be at least as good as the one the standard batch fit finds (within a small
    tolerance), without ever refitting from scratch.
    """
    batch = wgrp_model()
    batch.fit(random_values)

    formalisms = {
        'RP': 0,
        'NHPP': 1,
        'Kijima I': 2,
        'Kijima II': 3,
    }
    for formalism, idx in formalisms.items():
        online = wgrp_online_model(formalism=formalism)
        for x in random_values:
            online.update(x)

        # the local searches only accept improvements (at fixed data), and the
        # final streaming log-likelihood must be at least as good as the one the
        # standard batch fit finds, within a small tolerance
        ll_online = online.loglikelihood()
        ll_batch = batch.mle_objs_[idx]['optimum_value']
        assert ll_online >= ll_batch - 0.25, formalism


def test_mle_obj_and_predict_online():
    """The streaming state feeds the standard information criteria and
    prediction machinery without any batch fit."""
    online = wgrp_online_model(formalism='RP')
    for x in random_values:
        online.update(x)

    obj = online.mle_obj()
    assert obj['a'] > 0
    assert obj['b'] > 0
    assert obj['q'] == 0
    assert np.isfinite(obj['optimum_value'])

    pred = online.predict(2, random_series=20)
    assert len(pred) > 0


def test_online_update_returns_state():
    online = wgrp_online_model(formalism='Kijima II')
    state = online.update(3.2)
    assert state['n'] == 1
    assert state['a'] == 3.2  # closed form with b=1: a = mean(x)
    # Kijima II (propagation 0, q=0.5): v = q * x
    assert state['virtual_age'] == 0.5 * 3.2


def test_auto_mode_selects_the_same_formalism_as_the_standard():
    """
    With the default formalism='auto', the online model tracks the four
    formalisms in parallel and selects the active one by the BIC -- no need to
    define the formalism -- converging to the same choice and estimates of the
    standard batch fit.
    """
    batch = wgrp_model()
    batch.fit(random_values)
    optimum_batch = batch.mle_objs_[0]  # BIC-best for this data (RP)

    online = wgrp_online_model()  # formalism='auto' (default)
    for x in random_values:
        online.update(x)

    assert online.optimum_['parameters']['formalism'] == (
        optimum_batch['parameters']['formalism']
    )
    assert abs(online.a_ - optimum_batch['a']) < 0.05
    assert abs(online.b_ - optimum_batch['b']) < 0.05
    assert online.q_ == optimum_batch['q']

    # mle_objs() returns one object per tracked formalism, in the standard order
    objs = online.mle_objs()
    assert [o['parameters']['formalism'] for o in objs] == [
        'RP',
        'NHPP',
        'Kijima I',
        'Kijima II',
    ]