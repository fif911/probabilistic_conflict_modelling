# Learn the project

The purpose of this document is to provide a brief overview of the project structure and how to use it.

The installation process is described in the [README.md](README.md) file. Please follow it to set up the environment.

## Data Processing Pipeline

The data processing pipeline transforms the raw input data into a format suitable for conflict-related fatalities
forecasting.

Folder: `data_preprocessing_pipeline`.

The following steps outline the process:

### Step 1: Add Date to Conflict Data

The first step is to convert the `month_id` into an actual date. The `month_id` represents the number of months since
January 1980. We use this to create a proper date field, which is essential for temporal predictions.

**Script**: `1. add_date.py`

```bash
python 1. add_date.py
```

This script reads the `cm_features_v2.0.parquet` file and adds a new column, date, which corresponds to the first day of
the respective month. The output is saved as `cm_features_v2.1.csv`.

### Step 2: Add Country Codes

In this step, we ensure that each country in the dataset is assigned a unique country code from **CoW** (Correlates of
War)
dataset. This is done by
cross-referencing the conflict data with the **Gledish-Ward** country IDs, which were already provided by ViEWS.

**Script**: `2. add_ccode_to_cm_features.R`

```bash
Rscript 2. add_ccode_to_cm_features.R
```

This script reads the `cm_features_v2.1.csv` file, adds the necessary country codes, and outputs the updated dataset
as `cm_features_v2.2.csv`.

### Step 3: Remove Meaningless Data

Not all countries provide meaningful data for this analysis. This step removes countries with incomplete or
zero-fatality data, such as countries with a high percentage of missing values or consistently low conflict intensity.

**Script: 3**. `remove_meaningless.R`

```bash
Rscript 3. remove_meaningless.R
```

This script filters out countries with a high percentage of zero or missing values, ensuring the model works with
relevant data. The cleaned dataset is saved as `cm_features_v2.3.csv`.

### Step 4: Add Region Information

In this step, we add region-level data to each country, which allows the model to capture regional patterns and trends
in conflict data.

**Script: 4**. `add_region_to_cm.R`

```bash
Rscript 4. add_region_to_cm.R
```

The script appends regional information to the dataset and outputs the enhanced dataset as cm_features_v2.4.csv.

### Step 5: Create V-Dem Principal Components

The dataset contains a large number of variables from the V-Dem dataset, which may introduce noise and overfitting into
the model. To address this, we apply Principal Component Analysis (PCA) to reduce the dimensionality of the V-Dem
features while retaining the most important information.

**Script: 5**. `create_vdem_PCAs.py`

```bash
python 5. create_vdem_PCAs.py
```

This script performs PCA on the V-Dem features, reducing them to the top 11 components. The transformed dataset is saved
as cm_features_v2.5.csv.

### Step 6: Shift Yearly Country-Month Features

In the final step, we shift the features forward by one year, allowing the model to predict future conflict-related
fatalities based on past data.

**Script: 6**. `shift yearly cm_features.py`

```bash
python 6. shift\ yearly\ cm_features.py
```

This script shifts the features for future predictions and saves the final dataset as cm_features_v2.6.csv.

This concludes the data processing pipeline, and the resulting dataset is ready for model training and evaluation.
____

## Prediction Model

The prediction model uses the **NGBoost** framework to produce probabilistic forecasts of conflict-related fatalities.
This section outlines the model architecture, structure, and how to execute the model.

Folder: `model`.

### Purpose of Each File

1. **`run_model.py`**: This is the main control script for training and running the NGBoost model. It handles the entire
   model pipeline, including loading data, running feature selection, training the model, making predictions, and
   evaluating results. It also integrates **Optuna** for hyperparameter tuning to improve model performance.
2. **`NGBoost Regressor singlehorizon.py`**: This script is a development script for testing and debugging a single
   year (such as 2023). It focuses on fine-grained prediction analysis for research purposes.
3. **`least_important_features.py`**: This script defines which features are considered least important and removes them
   from the dataset before training. By eliminating these low-value features, the model becomes more efficient, reducing
   both computational load and the chance of overfitting.

#### Common Aspects Across Scripts

- **NGBoost** is used in all scripts as the primary model for probabilistic predictions.
- **Services** are used across the scripts for tasks like data loading, model training, and plotting.
- The dataset processed in the earlier steps (from the data pipeline) is used consistently in each script.

---

## Run the Model

**Run the model for multiple years** in by executing the `run_model.py` script:

```bash
python run_model.py
```

**Run the model for a single year with detailed analysis** by executing the `NGBoost Regressor singlehorizon.py` script:

```bash
python NGBoost\ Regressor\ singlehorizon.py
```

Note that you can also convert the `.py` files to `.ipynb` using Jupytext for easier visualization and debugging. This
process is automated using the bash scripts described in README.md.
___

## Model Services

The following services are designed to simplify various tasks in the modeling pipeline. They ensure that data loading,
transformations, training, evaluation, and saving predictions are handled consistently across different scripts.

### DataReadingService

**Purpose**: This service is responsible for reading input datasets and preparing them for use in model training. It
reads data from CSV and Parquet formats, processes columns, and handles missing values.

#### Key Functions:

- **read_features**: Loads the main feature dataset, processes target columns, and returns a clean DataFrame.
- **read_benchmark**: Reads benchmark models and aggregates the results for evaluation.

### DataSplittingService

**Purpose**: This service splits the dataset into training, validation, and test sets, ensuring that time-based splits
are handled correctly to avoid data leakage.

#### Key Functions:

- **split_by_time**: Splits the data into subsets based on time windows.
- **train_test_split**: Generates train and test splits for model evaluation.

### DataTransformationService

**Purpose**: Applies transformations to the data, such as log transformations and column manipulations. These
transformations help ensure the data is in the correct format for model training.

#### Key Functions:

- **log_transform**: Applies log transformation to continuous variables.
- **inverse_log_transform**: Reverts log transformation to original values.
- **pop_columns**: Removes specified columns from the dataset and returns them for later use.

### ModelCachingService

**Purpose**: This service manages the caching of trained models to disk. It ensures that models can be saved and reused
without needing retraining.

#### Key Functions:

- **cache_model**: Saves a trained model to disk.
- **get_model**: Loads a cached model from disk.
- **search_model_cache**: Searches for existing cached models in a directory.

### ModelEvaluationService

**Purpose**: Evaluates the performance of the model using various metrics like CRPScore. This service helps track how
well the model is performing and compare multiple models.

#### Key Functions:

- **evaluate**: Evaluates model performance based on various scoring metrics.
- **compare_models**: Compares the performance of different models and generates summary statistics.

### ModelTrainingService

**Purpose**: Handles the training process for the NGBoost model. It also manages hyperparameter tuning via Optuna to
optimize the model's configuration.

#### Key Functions:

- **train_model**: Trains the model based on a given configuration.
- **optimize_hyperparameters**: Runs Optuna trials to find the best hyperparameters.

### PlottingService

**Purpose**: Generates visualizations for interpreting the model’s performance and predictions. It includes plots for
SHAP values, prediction distributions, and more.

#### Key Functions:

- **plot_shap_values**: Visualizes feature importance using SHAP values.
- **predictions_plot**: Generates plots for predictions over time.
- **plot_cut_comparison**: Compares different strategies for handling negative predictions.

### PredictionSavingService

**Purpose**: Saves model predictions to disk in a structured format (e.g., Parquet). It ensures that predictions are
organized and accessible for further analysis or submission.

#### Key Functions:

- **save_predictions**: Saves predictions to a Parquet file.
- **load_predictions**: Loads saved predictions for evaluation.

___
In case something is unclear, or you need further assistance, please refer to the technical report linked in readme or
create an issue in the repository. I will be happy to help you with any questions or concerns.
