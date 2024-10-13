<img src="assets/logo.png" alt="LOGO" width="300" style="display: block; margin: auto;" />

# MESOR WGRP - PACKAGE

The `wgrp` package is a data science tool aimed at analyzing widespread renewal processes. Using an approach based on WGRP (Weibull-based renewal processes) [[1]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133772), the package allows you to study the behavior of systems that suffer some type of interference after certain occurrences. Although generally used for physical systems, WGRP can be applied to any system with these characteristics.

### Application Examples

- **Equipment Breakdown**: After a few breaks, the system tends to weaken. It is possible to model and predict future breakdown behavior.
- **Decrease in a Company's Shares**: If a company presents significant drops, even at random intervals, these drops can explain its future behavior.

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
- For more examples and advanced usage, see the Jupyter notebook available in the GitHub repository.
