# MESOR WGRP - PACKAGE

The `MWGRP` package is a data science tool aimed at analyzing widespread generalized renewal processes. Using an approach based on WGRP (Weibull-based renewal processes) [[1]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133772), the package allows one to study the behavior of systems exposed to interventions. Although generally used for technological systems, WGRP can be applied to any system on which interventions (e.g. preventive and corrective maintenance) might arise.

### Application Examples

- **Production Systems Breakdowns**: After registering when a few corrective and preventive interventions occurs, the  times between these interventions can be modeled via WGRP. It makes possible to evaluate the quality of the interventions as well as to predict when new interventions will be demanded.  Further, one can compare the performance of a number of systems via the respective WGRP models [[2]](https://www.sciencedirect.com/science/article/abs/pii/S0951832018308391).

- **Decrease in the value of a Company's Shares**: If a company presents relevant drops through the time, the times between drops can be modeled and thus forecasted via WGRP. Further, one can compare the finance performance of a number of companies via the respective WGRP models. 

- **Natural Catastrophic Events**: In the face of the history of when previous catastrophic events have occurred, one can model and forecast when new catastrophic events might occur. It is also possible to compare the natural condition between territories. 

## How to use

A Jupyter notebook with usage examples of most functions is available on [GitHub](https://github.com/danttis/mwgrp).

### Package Installation

To install the package, use the following command:

```bash
pip install mwgrp
```

### Import and Use of the `wgrp_model` Class

The `wgrp_model` class has `fit` and `predict` functions, which are similar to those available in other machine learning packages for ease of use.

```python
from mwgrp.model import wgrp_model
```

### Starting the Model with your Database

```python
# Initialize the model
model = wgrp_model()

# Example of failure data (time between failures - TBFs)
TBFs = [1, 2, 5]

# Fit the model to crash data
model.fit(TBFs) # See the function documentation for supported data types

# Make predictions
predict = model.predict(1)
```

---

### Additional Notes

- Be sure to consult the full documentation for additional details on the parameters and data types supported by the functions.
- For more examples and advanced usage, see the [Jupyter notebook](Example_of_use.ipynb) available in the GitHub repository.
