# Explainable Probabilistic Forecasting of Conflict-Related Fatalities

_There were not that many peaceful years in our history. Let's at least forecast when the next unrest comes._

___

## Motivation for the project

A new wave of violent conflicts around the world raises concerns about the security of the whole world. According to the
[ACLED Conflict Index](https://acleddata.com/conflict-index/) - 12% more conflicts occurred in 2023 compared to 2022 and
the trend does not seem to halt.

**The goal of our research is to build a robust model for early military conflict prediiction available to people.
Awareness of emerging risks should be a right, not a privilege.**

## What is already done

This repository presents the first publicly available and
explainable early conflict forecasting model capable of forecasting the distribution of conflict-related fatalities on a
country-month level. The model seeks to be maximally transparent and produces predictions
up to 14 months into the future. Our model improves over 4 out of 6 benchmark years but so far misses important violence
spikes.

**Keywords**: Interstate conflict modelling · Early Conflict Warning System
· Fatalities prediction · Predicting with uncertainty.

## Getting started

Before you proceed with running the model and iterating over existing implementations, it's important to understand the
inputs and outputs of the model.

The model takes as an **input** a dataset
from [ViEWS prediction competition 2024](https://viewsforecasting.org/research/prediction-challenge-2023/) augmented
with some additional features and PCAs. The full pipeline for data preprocessing is stored in
the `data_preprocessing_pipeline` folder.

The original dataset consists of UCDP Geo-referenced Event Dataset (GED), the V-Dem dataset, and the World
Development Indices, ACLED dataset and some others.

The model **outputs** the predicted distribution of **conflict-related fatalities** for each country-month pair (this is
a regression problem). The default prediction window is **14 months ahead**, as this was required by the ViEWS
competition rules. But this can be easily
adjusted in the `6. shift yearly cm_features.py` data pipeline file.

The great sources of information about the model are the technical report and the shortened version of the report. They
describe the model in detail and provide insights into the model's performance, as well as suggest possible
improvements.

### Technical report

The [technical report](https://drive.google.com/file/d/1r63S5BRPRl8G2HuTjyWtFpOxvVNsNV7o/view?usp=sharing) with details
of implementation and nuances of the model is available on the Google Drive.

For your convenience, the technical report structure is shown [below](#technical-report-structure).

### Shortened report

The [shortened version](https://medium.com/@zakotianskyi/predicting-wars-explainable-probabilistic-forecasting-of-conflict-related-fatalities-50c00cac02e4)
of the report is available on Medium. This report provides a high-level overview of the model and its performance.

## The Prediction Model

While the code is flexible and any model can be used, we build our model using the **Natural Gradient Boosting** (
NGBoost)
framework. The other models are in development.

The NGBoost model code is stored in the model folder in two representations: `.py` and `.ipynb` in the `model` folder.
To GitHub, we push only `.py` files. The `.ipynb` files are generated using Jupytext (see bash scripts in the section
below).

Simply run the script `.py` or `.ipynb` scripts, and it will train NGBoost model based on paramethers specified in the
header of the file, produce plots and submission files that can be evaluated to derive model accuracy.

### Model Evaluation

The model is evaluated using `evaluate_submissions.py` file, and the aggregated statistics about the model can be
gathered via `compare_submissions.ipynb`.

## For developers

### Install dependencies

Set up the environment using poetry by running the following command using base interpreter as **Python 3.10**:

```bash
poetry install
```

(install Poetry if you don't have it yet)

### Install pre-commit hooks

```bash
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

#### Generate jupyter notebooks based on .py files

Run the following command to generate jupyter notebook:

```bash
jupytext --to ipynb [file_name].py
```

Run the following command to turn jupyter notebook into a paired ipynb/py notebook:

```bash
jupytext --set-formats ipynb,py [file_name].ipynb
```

Run the following command to syncronize the jupyter notebook with changes in python file:

```bash
jupytext --sync [file_name].ipynb
```

### Technical report structure

The technical report is structured as follows:

1. Introduction
2. Related Work
3. Summary of contributions
4. Methodology
    1. Level of analysis and prediction window
    2. Original Competition Dataset
    3. Data preprocessing
        1. Data cleaning
        2. Dependent variable shifting
        3. Regions addition
        4. Parametrization
        5. Least Important Features Drop
    4. Natural Gradient Boosting
        1. Handling Negative Predictions
        2. Handling of removed countries
    5. Scoring Criteria
        1. Continuous Ranked Probability Score
        2. Ignorance Score
        3. Mean Interval Score
        4. Metrics Implementation
    6. Model fine-tuning
    7. Competition Benchmarks
        1. Last Historical Poisson
        2. Bootstraps from actuals
5. Results
    1. General Performance
    2. Additional evaluation for the 2022 year
    3. Model accuracy dependency on input fatalities distribution of
       the month
    4. Feature Importance
    5. Analysis of country forecasts
6. Discussion
7. Future work
8. Appedix with tables and figures

I hope you have fun reading it :P