# wgrp

**Weibull-based Generalized Renewal Processes (WGRP) in Python** — model, compare, and forecast the behavior of systems subject to interventions (e.g. preventive and corrective maintenance).

[![PyPI version](https://img.shields.io/pypi/v/wgrp.svg)](https://pypi.org/project/wgrp/)
[![Python versions](https://img.shields.io/pypi/pyversions/wgrp.svg)](https://pypi.org/project/wgrp/)
[![License](https://img.shields.io/pypi/l/wgrp.svg)](LICENSE)
[![Documentation](https://img.shields.io/readthedocs/wgrp?label=docs)](https://wgrp.readthedocs.io/en/latest/)

Maintained by the [MESOR](https://github.com/danttis/wgrp) research group (UFCA).

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick start (batch model)](#quick-start-batch-model)
- [Supported formalisms](#supported-formalisms)
- [Online (streaming) version](#online-streaming-version)
- [Documentation and examples](#documentation-and-examples)
- [Development](#development)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [License](#license)

## Overview

The `wgrp` package is a data science tool for analyzing generalized renewal processes. Based on the WGRP (Weibull-based Generalized Renewal Process) approach [[1]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133772), it allows one to study the behavior of systems exposed to interventions. Although generally used for technological systems, WGRP can be applied to any system on which interventions (e.g. preventive and corrective maintenance) might arise.

### Application examples

- **Production system breakdowns**: once the times of corrective and preventive interventions have been registered, the times between interventions can be modeled via WGRP. This makes it possible to evaluate the quality of the interventions as well as to predict when new interventions will be demanded. One can also compare the performance of several systems via their respective WGRP models [[2]](https://www.sciencedirect.com/science/article/abs/pii/S0951832018308391).

- **Natural catastrophic events**: given the history of when previous catastrophic events occurred, one can model and forecast when new catastrophic events might occur. It is also possible to compare the natural conditions between territories.

## Features

- **Batch estimation** (`wgrp_model`): fits the parameters `a`, `b`, and `q` of the WGRP model for **five maintenance formalisms in a single call**, and automatically selects the best one by information criteria at prediction time.
- **Prediction and plotting**: `n`-step-ahead forecasts built from bootstrap simulations, plus a comparison plot of the observed series and predicted quantiles.
- **Online (streaming) estimation** (`wgrp_online_model`): exact O(1) state updates per event, never refitting from scratch — see [below](#online-streaming-version).
- **Choice of optimizer**: particle swarm (`optimizer="ps"`, default) or dual simulated annealing (`optimizer="sa"`) for the global fit.
- **Standard data-science API**: `fit`/`predict`/`plot` methods, similar to those available in machine learning packages.

## Installation

```bash
pip install wgrp
```

Requires Python 3.11 or newer.

## Quick start (batch model)

```python
from wgrp.model import wgrp_model

# Initialize the model
model = wgrp_model()

# Failure data: times between failures (TBEs), cumulative failure times,
# or a DataFrame of intervention dates -- see the fit documentation
data = [1, 2, 5]

# Fit: estimates a, b, q for every formalism at once (results in model.mle_objs_)
model.fit(data)

# Predict: forecasts from the formalism selected by information criteria
predictions = model.predict(1)

# Plot: observed series vs. bootstrapped predictions and quantiles
model.plot()
```

`fit` accepts a `random_state` for reproducibility and an `optimizer` (`"ps"` for particle swarm, `"sa"` for simulated annealing). The per-formalism estimates are stored in `model.mle_objs_` (with information criteria), so you can compare formalisms yourself.

## Supported formalisms

`fit` estimates the model under different assumptions about the effect of each intervention, expressed by the propagation parameter `q`:

| Formalism | `q` | Interpretation |
|---|---|---|
| RP (Renewal Process) | 0 | Perfect repair: the system is as good as new after each intervention |
| NHPP (Non-Homogeneous Poisson Process) | 1 | Minimal repair: the system is as bad as old after each intervention |
| Kijima I | estimated | Imperfect repair whose restoration acts only on the most recent interarrival time (`v ← v + q·x`) |
| Kijima II | estimated | Imperfect repair whose restoration acts on the whole accumulated virtual age (`v ← q·(v + x)`) |
| Intervention type-based | estimated | Propagation depends on the type of intervention (preventive vs. corrective) |

Kijima I and Kijima II [[1]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133772) are useful for studying how partial or imperfect maintenance affects the time to the next failure: `q` between 0 and 1 interpolates between a perfect repair (RP) and a minimal repair (NHPP).

## Online (streaming) version

The `wgrp_online_model` class ingests the times between events (TBEs) one at a time and **never refits from scratch** over the whole series — just like the online linear regression update, in which running statistics are updated in O(1) per new observation:

- **Virtual age**: a Markov chain that depends only on the previous virtual age — O(1) per event;
- **Scale parameter `a`**: closed-form MLE `a = (S_n/n)^(1/b)` updated from a running sum `S` — O(1) per event and exactly equal to the value the standard package computes from scratch on the same data prefix;
- **Log-likelihood**: a sum of per-observation terms — O(1) per event;
- **`b` and `q`**: no closed-form estimator exists, so they are tracked by *warm-started local optimization* (bounded Nelder-Mead starting from the current `(b*, q*)` and accepting only improvements) — the classic recursive-estimation scheme, instead of the global swarm refit of the batch model. The estimates converge to the optimum of the log-likelihood over the data seen so far, at a fraction of the batch cost (in the windshield example, ~1s online vs ~35s batch);
- **Model selection**: with the default `formalism='auto'`, the online model tracks `'RP'`, `'NHPP'`, `'Kijima I'` and `'Kijima II'` in parallel and selects the active one by the BIC at every event — the same selection the standard package performs — so you never have to define the formalism yourself. Pass a specific name (e.g. `formalism='Kijima II'`) to track a single formalism.

```python
from wgrp.online import wgrp_online_model

# Online: one TBE at a time -- no formalism needed ('auto' by default)
model = wgrp_online_model()
model.update(1)   # O(1) exact state update + warm-started local search
model.update(2)
model.update(5)

model.a_, model.b_, model.q_        # streaming estimates of the BIC-best formalism
model.optimum_['parameters']['formalism']  # formalism selected so far
model.mle_obj()                     # standard model object (ICs, etc.)
model.predict(1)                    # predictions straight from the streaming state
```

See the [Update WGRP notebook](Update%20WGRP%20-%20code.ipynb) for the full comparison between the online and the standard models.

## Documentation and examples

- Full API documentation: [WGRP — Read the Docs](https://wgrp.readthedocs.io/en/latest/)
- General usage examples: [Example_of_use.ipynb](Example_of_use.ipynb)
- Online vs. batch comparison: [Update WGRP notebook](Update%20WGRP%20-%20code.ipynb)

## Development

The project dependencies are declared in `pyproject.toml` and the development environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
# create the virtualenv and install the package in editable mode
uv venv
uv pip install --python .venv/bin/python -e . pytest

# run the test suite (unit + doctests)
PYTHONPATH=. .venv/bin/python -m pytest

# serve the documentation locally (requires the mkdocs dependencies)
.venv/bin/python -m mkdocs serve
```

## Acknowledgements

We would like to thank the [National Council for Scientific and Technological Development (CNPq)](https://www.gov.br/cnpq/pt-br) and the [Federal University of Cariri (UFCA)](https://www.ufca.edu.br/) for granting the scholarships, and the Institutional Program for Scientific and Technological Initiation for all their support during the development of the project.

## Contact

If you have any questions about the package, its usage, or tips, feel free to contact the developers:

[Francisco Junior Peixoto Dantas](mailto:juniordante01@gmail.com)
[Paulo Renato Alves Firmino](mailto:paulo.firmino@ufca.edu.br)

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## References

1. Ferreira RJ, Firmino PRA, Cristino CT (2015). *A Mixed Kijima Model Using the Weibull-Based Generalized Renewal Processes*. PLoS ONE, 10(7), e0133772. https://doi.org/10.1371/journal.pone.0133772
2. de Oliveira CCF, Firmino PRA, Cristino CT (2019). *A tool for evaluating repairable systems based on Generalized Renewal Processes*. Reliability Engineering & System Safety, 183, 281–297. https://doi.org/10.1016/j.ress.2018.11.025