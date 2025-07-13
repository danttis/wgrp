# MESOR WGRP - PACKAGE

<!--
Here is a more developer-oriented version:
  1. In the terminal, install Poetry with the command: `pip install poet`.
  2. Navigate to the `wgrp` project folder and run: `poetry install` to install the dependencies.
  3. To generate the documentation, run: `poetry run task docs`.
  4. If you change any functions, run the tests with: `poetry run task test`.
-->

The `wgrp` package is a data science tool aimed at analyzing widespread generalized renewal processes. Using an approach based on WGRP (Weibull-based renewal processes) [[1]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133772), the package allows one to study the behavior of systems exposed to interventions. Although generally used for technological systems, WGRP can be applied to any system on which interventions (e.g. preventive and corrective maintenance) might arise.

### Application Examples

- **Production Systems Breakdowns**: After registering when a few corrective and preventive interventions occurs, the  times between these interventions can be modeled via WGRP. It makes possible to evaluate the quality of the interventions as well as to predict when new interventions will be demanded.  Further, one can compare the performance of a number of systems via the respective WGRP models [[2]](https://www.sciencedirect.com/science/article/abs/pii/S0951832018308391).

- **Natural Catastrophic Events**: In the face of the history of when previous catastrophic events have occurred, one can model and forecast when new catastrophic events might occur. It is also possible to compare the natural condition between territories. 


Furthermore, the package supports the consideration of different assumptions about the effect of maintenance through the Kijima I and II models, which represent, respectively:

- Kijima I: where the degree of restoration depends only on the time since the last intervention;
- Kijima II: where the cumulative effect of interventions is considered, reflecting more realistic scenarios in complex systems.

These models are useful for studying how partial or imperfect maintenance affects the time to next failure.

## How to use

A Jupyter notebook with usage examples of most functions is available on [GitHub](https://github.com/danttis/wgrp).

### Package Installation

To install the package, use the following command:

```bash
pip install wgrp
```

### Import and Use of the `wgrp_model` Class

The `wgrp_model` class has `fit` and `predict` functions, which are similar to those available in other machine learning packages for ease of use.

```python
from wgrp.model import wgrp_model
```

### Starting the Model with your Database

```python
# Initialize the model
model = wgrp_model()

# Example of failure data (time between failures)
data = [1, 2, 5]

# Fit the model to crash data
model.fit(data) # See the function documentation for supported data types

# Make predictions
predict = model.predict(1)
```
See more about function documentation at: [WGRP - Read the Docs](https://wgrp.readthedocs.io/en/latest/)

---


### Additional Notes

- Be sure to consult the full documentation for additional details on the parameters and data types supported by the functions.
- For more examples and advanced usage, see the [Jupyter notebook](https://github.com/danttis/wgrp/blob/main/Example_of_use.ipynb) available in the GitHub repository.

If you have any questions about the package, its usage, or tips, feel free to contact the developers:  
[Francisco Junior Peixoto Dantas](mailto:juniordante01@gmail.com)  
[Paulo Renato Alves Firmino](mailto:paulo.firmino@ufca.edu.br)

