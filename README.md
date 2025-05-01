# ml_preprocessor
Author: Owen Rasor

A reusable, pipeline-compatible data preprocessing tool built with **pandas** and **scikit-learn** to prepare datasets for machine learning workflows.

---

## Features

- Cleans missing values (target-aware)
- Detects and classifies column types:
  - Numerical
  - Categorical
  - Datetime
  - Boolean
- Normalizes numeric data (StandardScaler)
- Encodes categorical features (OneHotEncoder)
- Handles outliers (IQR method)
- Engineers datetime features (year, month, day, weekday)
- Splits dataset into **train**, **validation**, and **test**
- Compatible with scikit-learn pipelines
- Supports saving/loading preprocessor with `joblib`
- Built-in logging for debugging and reproducibility

