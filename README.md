# Explainable Probabilistic Forecasting of Conflict-Related Fatalities

Can we predict wars? How certain would we be in our predictions?
This research presents the first publicly available and explainable
early conflict forecasting model capable of forecasting the distribution
of conflict-related fatalities on a country-month level. The model seeks
to be maximally transparent, uses publicly available data and produces
predictions up to 14 months into the future. Our model improves over
3 out of 4 benchmark years but misses the violence spikes, possibly, due
to the nature of independent variables in the dataset. The presented
model can complement the field of conflict-warning systems and serve as
a reference against which future improvements can be evaluated.

**Keywords**: Interstate conflict modelling · Early Conflict Warning System
· Fatalities prediction · Predicting with uncertainty.

The report with details is
available [here](https://drive.google.com/file/d/1r63S5BRPRl8G2HuTjyWtFpOxvVNsNV7o/view?usp=sharing).

The shortened version of the report is available
at [Medium](https://medium.com/@zakotianskyi/predicting-wars-explainable-probabilistic-forecasting-of-conflict-related-fatalities-50c00cac02e4).

## Model

The NGBoost model code is stored in the model folder in two representations: `.py` and `.ipynb`. Simply run the script,
and it will produce plots along with submission files.

### Model Evaluation

The model is evaluated using `evaluate_submissions.py` file, and the aggregated statistics about the model can be
gathered via `compare_submissions.ipynb`.

## For developers

### Install dependencies

Set up the environment using poetry by running the following command:

```bash
poetry install
```

### Install pre-commit hooks

Run the following command to install pre-commit hooks:

```bash
pre-commit install
```

Ensure that you have the following dependencies installed:

1) Black (for Python code formatting)
2) Jupyter (for removing output from notebooks)

### Jupytext

For better development experience and version control, the Jupytext library is used to generate `.py` files based on
their `.ipynb` representation and vice-versa. Additionally, Jupytext provides a convenient syncing logic between both
representations.

### Bash scripts

There are two bash scripts available:

- **data_preprocessing_pipeline.sh** - script for running all steps of the data preprocessing pipeline. Note that this
  requires an R and Python environment set-up. The reason for this is that the pipeline uses some libraries exclusively
  available in R only.
- **jupytext_sync.sh** - script to create a Jupyter model file and sync it with its Python representation.

Run the following command to give execute permission to bash script:

```bash
chmod +x [file].sh
```

Run the following command to execute the bash script:

```bash
./[file].sh [args]
```
