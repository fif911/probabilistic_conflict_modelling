# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Imports
#

# +
import os
import warnings
import pickle
import glob
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from ngboost.scores import CRPScore, LogScore
from ngboost.distns import Poisson, Normal, MultivariateNormal
from ngboost import NGBRegressor

from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
import CRPS.CRPS as pscore
import shap

from utilities.views_utils import views_month_id_to_date

# -

# # Variables

# +
# Input data
prediction_year: int = 2024
prediction_window: int = 14
cm_features_version: str = "2.5"

# Run config
SHOW_PLOTS: bool = True
PLOT_STD: bool = True
SAVE_FIGURES: bool = True
USE_CACHED_MODEL: bool = True
SAVE_PREDICTIONS: bool = True

# Data preparation settings
INCLUDE_COUNTRY_ID: bool = True
INCLUDE_MONTH_ID: bool = True
DROP_0_ROWS_PERCENT: int = 20
DROP_35_LEAST_IMPORTANT: bool = False
LOG_TRANSFORM: bool = False

# Model settings
dist = Normal
score = CRPScore
n_estimators: int = 300
bs_max_depth: int = 5
minibatch_frac: float = 0.5
base_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=bs_max_depth,
    splitter="best",
    random_state=42,
)
# -

# # Paths

# +
folder_to_str: str = (
    f"ng_boost_cm_v{cm_features_version}_"
    f"pw_{prediction_window}_"
    f"{dist.__name__.lower()}_"
    f"d_{DROP_0_ROWS_PERCENT}_"
    f"n_{n_estimators}_"
    f"s_{score.__name__.lower()}_"
    f"c_{str(INCLUDE_COUNTRY_ID)[0]}_"
    f"m_{str(INCLUDE_MONTH_ID)[0]}_"
    f"bsd_{bs_max_depth}_"
    f"mbf_{minibatch_frac}_"
    f"dli_{35 if DROP_35_LEAST_IMPORTANT else 0}_"
    f"log_{str(LOG_TRANSFORM)[0]}"
)

model_cache_prefix: str = f"../model_cache/{folder_to_str}/window=Y{prediction_year}"
figures_prefix: str = f"../figures/{folder_to_str}/window=Y{prediction_year}"
submission_prefix: str = f"../submission/{folder_to_str}"
print("Model coding: ", folder_to_str)

month_to_date = lambda x: f"{1980 + (x - 1) // 12}-{((x - 1) % 12) + 1}-01"

# -

os.makedirs(f"{figures_prefix}/force_plots", exist_ok=True)
os.makedirs(f"{figures_prefix}/histograms", exist_ok=True)

# # Data Preparation

cm_features = pd.read_csv(
    f"../data/cm_features_v{cm_features_version}_Y{prediction_year}.csv"
)

# +
# load benchmark model
model_names: dict[str, str] = {
    "bootstrap": "bm_cm_bootstrap_expanded_",
    "poisson": "bm_cm_last_historical_poisson_expanded_",
    "conflictology": "bm_conflictology_cm_",
}
model_name: str = model_names["poisson"]
benchmark_model = pd.read_parquet(
    f"../benchmarks/{model_name}{prediction_year}.parquet"
)

# Group by 'month_id' and 'country_id' and calculate mean and std for each group
agg_funcs: dict[str, tuple[str, str]] = {
    "outcome": (
        "mean",
        "std",
    )  # Assuming 'prediction' is the column to aggregate; adjust if necessary
}

# there is 20 draws per each country per each month. Get the mean of the draws and std for each month
benchmark_model = (
    benchmark_model.groupby(["month_id", "country_id"]).agg(agg_funcs).reset_index()
)

# Flatten the multi-level columns resulting from aggregation
benchmark_model.columns = [
    "_".join(col).strip() if col[1] else col[0]
    for col in benchmark_model.columns.values
]

# Rename columns
benchmark_model.rename(
    columns={"outcome_mean": "outcome", "outcome_std": "outcome_std"}, inplace=True
)

if prediction_year == 2024:
    # SHIFT MONTHS BACK BY 7 MONTHS in benchmarks as it starts from.
    # TODO: Check if this is valid (probably not but this is a bug in inputs)
    # TODO: Report to PRIO
    benchmark_model["month_id"] = benchmark_model["month_id"] - 6

# add date column
benchmark_model["date"] = views_month_id_to_date(benchmark_model["month_id"])
print([month_to_date(month) for month in benchmark_model["month_id"].unique()])

# benchmark_model.head(5)

# +
# load actuals
actuals_model = pd.read_parquet(
    f"../actuals/cm/window=Y{prediction_year}/cm_actuals_{prediction_year}.parquet"
).reset_index(drop=False)

# actuals_model = actuals_model.groupby(['month_id', 'country_id']).mean().reset_index()
actuals_model["date"] = views_month_id_to_date(actuals_model["month_id"])
print(actuals_model["month_id"].unique())

# rename outcome to ged_sb
actuals_model.rename(columns={"outcome": "ged_sb"}, inplace=True)

# +
# get all columns that have ged_sb_y_ in the name
all_targets: list[str] = [
    col for col in cm_features.columns if col.startswith("ged_sb_y_")
]
target: str = f"ged_sb_{prediction_window}"

# drop all possible targets except the chosen
try:
    all_targets.remove(target)
except ValueError:
    pass
cm_features.drop(columns=all_targets, inplace=True)

# drop all rows for which ged_sb_y_15 is NAN
cm_features = cm_features.dropna()

if SHOW_PLOTS:
    # plot target per month
    cm_features[target].plot()
    cm_features["ged_sb"].plot()
    cm_features["ged_sb_tlag_6"].plot()
    plt.legend()
    plt.show()

# +
cm_features = cm_features.drop(columns=["country", "gleditsch_ward"], errors="ignore")
# drop if exists 'year', 'ccode'
cm_features = cm_features.drop(
    columns=["year", "ccode", "region", "region23"], errors="ignore"
)

LEAST_IMPORTANT_FEATURES: list[str] = [
    "general_efficiency_t48",
    "wdi_sp_dyn_imrt_in",
    "vdem_v2x_egal",
    "vdem_v2x_partipdem",
    "vdem_v2x_partip",
    "vdem_v2x_libdem",
    "dam_cap_pcap_t48",
    "vdem_v2xdd_dd",
    "vdem_v2x_edcomp_thick",
    "groundwater_export_t48",
    "wdi_sh_sta_stnt_zs",
    "region_Middle East & North Africa",
    "vdem_v2x_execorr",
    "region23_Western Asia",
    "region23_Southern Europe",
    "region23_Northern Africa",
    "region_Sub-Saharan Africa",
    "region23_Caribbean",
    "region23_Eastern Europe",
    "region23_Eastern Africa",
    "region23_South-Eastern Asia",
    "region23_Middle Africa",
    "region23_Northern Europe",
    "region23_Western Africa",
    "region23_Southern Africa",
    "region23_South America",
    "region_Latin America & Caribbean",
    "region23_Northern America",
    "region_North America",
    "region23_Melanesia",
    "region23_Eastern Asia",
    "region23_Central Asia",
    "region23_Central America",
    "region_Europe & Central Asia",
    "region23_Western Europe",
]

if DROP_35_LEAST_IMPORTANT:
    print("Current number of features:", len(cm_features.columns))
    cm_features = cm_features.drop(columns=LEAST_IMPORTANT_FEATURES)
    print(
        "Number of features after dropping 35 least important:",
        len(cm_features.columns),
    )

# cm_features = cm_features.drop(
#     columns=['ged_sb_tlag_2', 'ged_sb_tlag_3', 'ged_sb_tlag_4', 'ged_sb_tlag_5', 'ged_sb_tlag_1', 'ged_sb_tlag_6', ])
# # drop ged_sb, ged_ns, ged_os, acled_sb, acled_sb_ count, acled_os, ged_sb_tsum_24
# cm_features = cm_features.drop(
#     columns=['ged_sb', 'ged_ns', 'ged_os', 'acled_sb', 'acled_sb_count', 'acled_os', 'ged_sb_tsum_24', 'ged_os_tlag_1'])
# # drop splag_1_decay_ged_sb_5, splag_1_decay_ged_os_5, splag_1_decay_ged_ns_5, decay_ged_sb_5, decay_ged_os_5, decay_ged_sb_500, decay_ged_os_100, decay_ged_ns_5, decay_ged_ns_100, decay_acled_sb_5, decay_acled_os_5, decay_acled_ns_5
# cm_features = cm_features.drop(
#     columns=['splag_1_decay_ged_sb_5', 'splag_1_decay_ged_os_5', 'splag_1_decay_ged_ns_5', 'decay_ged_sb_5',
#              'decay_ged_os_5', 'decay_ged_sb_500', 'decay_ged_os_100', 'decay_ged_ns_5', 'decay_ged_ns_100',
#              'decay_acled_sb_5', 'decay_acled_os_5', 'decay_acled_ns_5', 'decay_ged_sb_100'])

# +
# save first 50 rows of the dataset to a new csv file with column names
# cm_features.head(50).to_csv('data/cm_features_first_50.csv', index=True)
# cm_features.head(10)
# prepare dataset for machine learning

cm_features["date"] = pd.to_datetime(cm_features["date"])
cm_features["country_id"] = cm_features["country_id"].astype("category")

cm_features.head(5)
# -

# One-hot encode 'country_id'
if INCLUDE_COUNTRY_ID:
    # TODO: try what changes if encode ccode
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(cm_features[["country_id"]])
    countries_encoded = encoder.transform(cm_features[["country_id"]])

    # rename the columns
    countries_encoded = pd.DataFrame(
        countries_encoded, columns=encoder.get_feature_names_out(["country_id"])
    )
    countries_encoded = countries_encoded.drop(
        columns="country_id_1"
    )  # drop country_id_1
    # drop na

    # countries_encoded
    # merge the encoded features with the original dataset
    cm_features = pd.concat([cm_features, countries_encoded], axis=1)
    cm_features = cm_features.dropna()

if LOG_TRANSFORM:
    cm_features_original_target = cm_features[target]
    cm_features[target] = np.log(cm_features[target] + 1)

# +
# Split the dataset into training and test sets
# prediction_year = 2018
# test(final model evaluation): Jan 2018 - Dec 2018 (12 months)
# Nov 2016 (Nov 2016 month_id 443)    predicts Jan 2018 (Nov 2016 443+3+11=457)
# Oct 2017 (Nov 2016 month_id+11=454) predicts Dec 2018 (Oct 2017 454+3+11=468)
# train_df is until Oct 2016 inclusive
# test_df is one year from Nov 2016 to Oct 2017 inclusive
# Note: Ideally:
# Oct 2017 454 predicts Jan 2018 457
# Oct 2017 454 predicts Feb 2018 458
# ...
# Oct 2017 454 predicts Dec 2018 468
last_month_id: int = cm_features["month_id"].max()

train_features_to_oct: int = last_month_id - 11 - 3
test_features_since_nov: int = last_month_id - 11

print("features_to_oct:", train_features_to_oct)
print("features_since_nov:", test_features_since_nov)

train_df = cm_features[
    cm_features["month_id"] <= train_features_to_oct
]  # train is till 476 inclusive

# test_df is one year from Nov to Oct inclusive (479-490)
test_df = cm_features[(cm_features["month_id"] >= test_features_since_nov)]

# +
# print(train_df['month_id'].unique())
# print(test_df['month_id'].unique())
# # count number of unique dates
# print(test_df['month_id'].nunique())

# +
# count number of rows where target is 0
# drop 0 rows from train df
if DROP_0_ROWS_PERCENT > 0:
    print(f"Initial count: {train_df[train_df[target] == 0].shape[0]}")

    indices = train_df[train_df[target] == 0].index.to_series()

    num_to_drop: int = int(len(indices) * DROP_0_ROWS_PERCENT / 100)
    indices_to_drop = indices.sample(n=num_to_drop, random_state=42)

    train_df = train_df.drop(indices_to_drop)

    print(f"Count after removal: {train_df[train_df[target] == 0].shape[0]}")

test_df.reset_index(drop=True, inplace=True)

# +
# shuffle the training set
# train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# +
# save date column for test_df
test_df_date = test_df["date"]
train_df_date = train_df["date"]

test_df_country_name = test_df["gw_statename"]
train_df_country_name = train_df["gw_statename"]

train_df_country_id = train_df["country_id"]
test_df_country_id = test_df["country_id"]

train_df_month_id = train_df["month_id"]
test_df_month_id = test_df["month_id"]

test_df = test_df.drop("date", axis=1)
test_df = test_df.drop("country_id", axis=1)
test_df = test_df.drop("gw_statename", axis=1)

train_df = train_df.drop("date", axis=1)
train_df = train_df.drop("country_id", axis=1)
train_df = train_df.drop("gw_statename", axis=1)

# if CREATE_VAL_DS:
#     val_df_date = validation_df['date']
#     val_df_country_id = validation_df['country_id']
#     val_df_month_id = validation_df['month_id']
#     validation_df = validation_df.drop('date', axis=1)
#     validation_df = validation_df.drop("country_id", axis=1)
#     validation_df = validation_df.drop("gw_statename", axis=1)

if not INCLUDE_MONTH_ID:
    test_df = test_df.drop("month_id", axis=1)
    train_df = train_df.drop("month_id", axis=1)
    # if CREATE_VAL_DS:
    #     validation_df = validation_df.drop('month_id', axis=1)

print(test_df_month_id.unique())

print("Difference between benchmark and test month_id:")
print(benchmark_model["month_id"].min() - test_df_month_id.max())
print(benchmark_model["month_id"].min() - test_df_month_id.min())
# train_df.head(5)
# -

# # Train Test Split

# +
X_train = train_df.drop(target, axis=1)
y_train = train_df[target]

X_test = test_df.drop(target, axis=1)
y_test = test_df[target]

# if CREATE_VAL_DS:
#     X_val = validation_df.drop(target, axis=1)
#     y_val = validation_df[target]

# -

#
# # Model

# +
import warnings
from datetime import datetime
from pathlib import Path
import pickle
import glob

# Model tuning:
model_files = list(Path(model_cache_prefix).glob("model_*.p"))

cached_model_available = len(model_files) > 0
print("Cached model available:", cached_model_available)

if USE_CACHED_MODEL and cached_model_available:
    model_path = model_files[0]

    with model_path.open("rb") as file:
        ngb = pickle.load(file)
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"Training NGB with {n_estimators} estimators and {score} score...")
        ngb = NGBRegressor(
            n_estimators=n_estimators,
            verbose_eval=10,
            Dist=dist,
            learning_rate=0.01,
            Score=score,
            random_state=42,
            Base=base_learner,
            minibatch_frac=minibatch_frac,
            # col_sample=1.0,
        ).fit(
            X_train, y_train, X_test, y_test
        )  # be careful with this, not to use early stopping

    print("Done!")

    for file_path in model_files:
        file_path.unlink()
    Path(model_cache_prefix).mkdir(parents=True, exist_ok=True)

    time = datetime.now()

    model = (
        "model_" f"{time.day}_{time.month}_{time.year}_" f"{time.hour}_{time.minute:02}"
    )

    with Path(model_cache_prefix, model).with_suffix(".p").open("wb") as file:
        pickle.dump(ngb, file)

# -

# # Predictions

# +
ngb_train_predictions = ngb.predict(X_train)
ngb_predictions = ngb.predict(X_test)
ngb_predictions_dist = ngb.pred_dist(X_test)
# means and stds of the predictions
# mean, std = ngb_predictions_dist.loc, ngb_predictions_dist.scale

if not LOG_TRANSFORM:
    ngb_train_predictions = [max(0, pred) for pred in ngb_train_predictions]
    ngb_predictions = [max(0, pred) for pred in ngb_predictions]

sampled_dist = ngb_predictions_dist.sample(1000).T

# -

# # Insights


# +
def calculate_CRPS(sampled_dist, y_true, index_start, index_end):
    scores: list[float] = []

    for indx, y_point in enumerate(y_true[index_start:index_end]):
        crps_score: float = pscore(sampled_dist[indx], y_point).compute()[0]
        scores.append(crps_score)

    return np.average(scores)


# average crps score

print(f"Average CRPS score: {calculate_CRPS(sampled_dist, y_test, 0, len(y_test))}")

# +
# map predictions to months based on the test_df
test_df["ngb_predictions"] = ngb_predictions
train_df["ngb_predictions"] = ngb_train_predictions

# add date column back to test_df and add to each date shift of 14 months
test_df["date"] = test_df_date + pd.DateOffset(months=prediction_window)
train_df["date"] = train_df_date

test_df["country_id"] = test_df_country_id
train_df["country_id"] = train_df_country_id

test_df["month_id"] = test_df_month_id
train_df["month_id"] = train_df_month_id

test_df["country_name"] = test_df_country_name
train_df["country_name"] = train_df_country_name

# +
# TODO: Improve metrics and use all metrics from the VIEWS competition
# Calculate RMSE
# train_rmse = sqrt(mean_squared_error(y_train, xg_lss_pred_train))
# actuals_rmse = sqrt(mean_squared_error(actuals_model['ged_sb'], predictions))
# benchmark_rmse = sqrt(mean_squared_error(y_test, benchmark_model['outcome']))
ngb_train_rmse = sqrt(mean_squared_error(y_train, ngb_train_predictions))
ngb_test_rmse = sqrt(mean_squared_error(y_test, ngb_predictions))
all_zeros_rmse = sqrt(mean_squared_error(y_test, [0] * len(y_test)))
# added if cases because length need to be equal, and if year is not full,
# then we need to make benchmark_model['outcome'] and actuals_model['ged_sb']

if prediction_year == 2024:
    actuals_length = len(actuals_model["ged_sb"])
    actuals_bench_rmse = sqrt(
        mean_squared_error(
            actuals_model["ged_sb"], benchmark_model["outcome"][:actuals_length]
        )
    )
else:
    actuals_bench_rmse = sqrt(
        mean_squared_error(actuals_model["ged_sb"], benchmark_model["outcome"])
    )

print("Cm features version:", cm_features_version)
print(f"Prediction year: {prediction_year}")
print(f"Include country_id: {INCLUDE_COUNTRY_ID}")
print(f"Include month_id: {INCLUDE_MONTH_ID}")
print(f"Drop train 0 rows: {DROP_0_ROWS_PERCENT}%")
print(f"Normal distribution: {dist == Normal}")
print(f"Number of estimators: {n_estimators}")
print(f"Score: {str(score.__name__)}")
print(f"Log transform: {LOG_TRANSFORM}")
print(f"Base learner max depth: {bs_max_depth}")
print(f"Minibatch fraction: {minibatch_frac}")

# print(f"XGB [train predictions] RMSE: {train_rmse}")
# print(f"XGB [test predictions]  RMSE YTEST VS PREDICTIONS: {rmse}")
print(f"\nNGB [train predictions] RMSE NGB: {ngb_train_rmse}")
print(f"NGB [test predictions]  RMSE NGB: {ngb_test_rmse}")
# if CREATE_VAL_DS:
#     ngb_val_rmse = sqrt(mean_squared_error(y_val, ngb.predict(X_val)))
#     print(f"NGB [validation predictions] RMSE NGB: {ngb_val_rmse}")

# print(f"RMSE YTEST VS ACTUALS: {actuals_rmse}')
# print(f"RMSE YTEST VS BENCHMARK: {benchmark_rmse}')
print(f"All Zeros: {all_zeros_rmse}")
print(f"\nBenchmark: RMSE ACTUALS VS BENCHMARK: {actuals_bench_rmse}")
# -

if LOG_TRANSFORM:
    test_df["original_" + target] = cm_features_original_target
    test_df["ngb_predictions_inverselog_std"] = (np.exp(sampled_dist) - 1).std(axis=1)
    test_df["ngb_predictions_inverselog"] = np.exp(test_df["ngb_predictions"]) - 1

if PLOT_STD:
    # dir(ngb.pred_dist(X_test).scale)
    # ngb.pred_dist(X_test).
    # save std of the predictions
    # TODO: Keep only sampled and remove normal_enabled condition

    ngb_predictions_std = sampled_dist.std(axis=1)
    ngb_predictions_std_ = np.sqrt(ngb_predictions_dist.var)

    confidence: float = 0.10
    if not (
        sum(ngb_predictions_std <= ngb_predictions_std_ * (1 - confidence))
    ) and not (sum(ngb_predictions_std >= ngb_predictions_std_ * (1 + confidence))):
        print(f"Confidence: +-{confidence:.0%}")
    # ngb_predictions_std_[0] # float
    # if dist == Normal:
    #     ngb_predictions_std = np.sqrt(ngb_predictions_dist.var)
    # else:
    #     # sampled_dist = ngb_predictions_dist.sample(1000)
    #
    #     ngb_predictions_max = sampled_dist.max(axis=0)
    #     ngb_predictions_min = sampled_dist.min(axis=0)
    #
    #     test_df['ngb_predictions_max'] = ngb_predictions_max
    #     test_df['ngb_predictions_min'] = ngb_predictions_min

    # add std to test_df
    test_df["ngb_predictions_std"] = ngb_predictions_std

# +
# temp = test_df
# temp.reset_index(inplace=False, drop=False)
#
# # get row with the highest number of deaths
# temp[temp['predictions'] == temp['predictions'].max()]
# print(temp[temp['predictions'] == temp['predictions'].min()])

cutoff_date = actuals_model["date"].iloc[-1].date().strftime("%Y-%m-%d")

if prediction_year == 2024:
    cutoff_date = pd.to_datetime(f"{cutoff_date}")

else:
    cutoff_date = pd.to_datetime(f"{prediction_year}-12-01")

# NOTE !!!!
# test_df.reset_index(inplace=True, drop=True)
ngb_predictions_sampled = ngb_predictions_dist.sample(1000).T.astype(int)
test_df_edge = test_df.shape[1]
test_df_new = pd.concat([test_df, pd.DataFrame(ngb_predictions_sampled)], axis=1)

ngb_predictions_sampled[12] == test_df_new.iloc[12, test_df_edge:]

# get the row with the highest number of deaths
# actuals_model


# add to test_df_new the actuals based on month_id and country_id
# actuals_model.rename(columns={'ged_sb': 'actuals'}, inplace=True)
# test_df_new.merge(actuals_model[['month_id', 'country_id', 'actuals']], on=['month_id', 'country_id'])

# drop level 0 and index columns
# test_df.drop(columns=['level_0', 'index'], inplace=True, errors='ignore')# test_df_new
# -

if SAVE_PREDICTIONS:
    # Save predictions
    # TODO: for countries that are in actuals but not in the predictions, add them to the predictions with 0
    #  test_df['country_id'].unique()
    #  actuals_model['country_id'].unique()
    missing_countries = set(benchmark_model["country_id"].unique()) - set(
        test_df_new["country_id"].unique()
    )

    # save predictions to a csv file
    # for each month for each country create 20 draws of the prediction named outcome
    # the structure of the file should be month_id, country_id, draw, outcome
    new_predictions_list = []
    all_countries = set(test_df_new["country_id"].unique()).union(missing_countries)

    for month_id in test_df_new["month_id"].unique():
        for country_id in all_countries:
            this_country_month = test_df_new[
                (test_df_new["month_id"] == month_id)
                & (test_df_new["country_id"] == country_id)
            ]

            if country_id in missing_countries:
                outcomes = np.zeros(1000)
            else:
                outcomes = this_country_month.iloc[:, test_df_edge:].values[0]

                if LOG_TRANSFORM:
                    outcomes = np.exp(outcomes) - 1

                # remove all values smaller than 0
                non_negatives = outcomes[outcomes >= 0]
                negative_counts = np.sum(outcomes < 0)

                if negative_counts > 0:
                    # Sample from the non-negative distribution to replace negative values
                    # We assume the distribution of non-negatives is suitable for sampling
                    sampled_values = np.random.choice(
                        non_negatives, size=negative_counts
                    )
                    outcomes[outcomes < 0] = sampled_values

                assert all(
                    outcomes >= 0
                ), "There are still negative values in the outcomes"

            new_predictions_list.extend(
                [
                    {
                        "month_id": month_id
                        + prediction_window,  # adjust for prediction window
                        "country_id": country_id,
                        "draw": draw,
                        "outcome": outcome,
                    }
                    for draw, outcome in enumerate(outcomes, start=0)
                ]
            )

    # set month_id, country_id, draw as int and outcome as float
    new_predictions = pd.DataFrame(new_predictions_list)

    new_predictions["month_id"] = new_predictions["month_id"].astype(int)
    new_predictions["country_id"] = new_predictions["country_id"].astype(int)
    new_predictions["draw"] = new_predictions["draw"].astype(int)
    new_predictions["outcome"] = new_predictions["outcome"].astype(int)

    # set index to month_id, country_id, draw
    new_predictions.set_index(["month_id", "country_id", "draw"], inplace=True)

    # create folder if it does not exist recursively

    folder = f"{submission_prefix}/cm/window=Y{prediction_year}"
    os.makedirs(folder, exist_ok=True)

    new_predictions.to_parquet(folder + f"/submission_Y{prediction_year}.parquet")

    print(f"Predictions saved")
    print(f"Saved to {folder}")
else:
    print("Predictions not saved")

# +
# Load data
country_list = pd.read_csv("../data/country_list.csv")
country_ids = test_df["country_id"].unique().tolist()

# Settings
num_plots_per_figure = 4
plt.figure(figsize=(15, 10))  # New figure
plots_added = 0

# Continue looping until all countries have been considered
max_date_train = pd.to_datetime(train_df["date"].max())
min_date_test = pd.to_datetime(test_df["date"].min())

# 1 year buffer because of validation set
expected_min_date_test = max_date_train + relativedelta(
    years=0, months=prediction_window + 3
)  # 15 is window size + 1 is from Sep to Oct

print(f"Max date in training set: {max_date_train}")
print(f"Min date in test set: {min_date_test}")
print(f"Expected min date in test set: {expected_min_date_test}")

# assert the different is exactly 15 months
assert min_date_test == expected_min_date_test

for index, country_id in enumerate(country_ids):
    # Ensure we output actual values only till they exist for 2024 edge case

    this_country_test = test_df[test_df["country_id"] == country_id]
    this_country_train = train_df[train_df["country_id"] == country_id]
    this_country_train = this_country_train.tail(24)
    if LOG_TRANSFORM:
        this_country_target = this_country_test["original_" + target][
            this_country_test["date"] <= cutoff_date
        ]
    else:
        this_country_target = this_country_test[target][
            this_country_test["date"] <= cutoff_date
        ]

    predictions_vector = (
        this_country_test["ngb_predictions"]
        if not LOG_TRANSFORM
        else this_country_test["ngb_predictions_inverselog"]
    )
    std_vector = (
        this_country_test["ngb_predictions_std"]
        if not LOG_TRANSFORM
        else this_country_test["ngb_predictions_inverselog_std"]
    )

    country_name = country_list[country_list["country_id"] == country_id][
        "name"
    ].values[0]

    # Check if country should be skipped because it is not interesting
    if this_country_target.sum() == 0:
        if SHOW_PLOTS:
            print(f"Skipping {country_name} as all actual are 0")
        continue

    # Prepare the subplot for non-skipped countries
    plt.subplot(2, 2, plots_added + 1)

    # Plotting data
    plt.plot(
        this_country_train["date"],
        this_country_train["ged_sb"],
        label="Train",
        color="gray",
        linestyle="-",
        marker="",
    )
    plt.plot(
        this_country_test["date"].apply(lambda x: x - relativedelta(months=14)),
        this_country_test["ged_sb"],
        label=f"Test Input",
        color="black",
        linestyle="--",
        marker="",
    )

    plt.plot(
        this_country_test["date"][this_country_test["date"] <= cutoff_date],
        this_country_target,
        label="Actual",
        color="black",
        linestyle="-",
        marker="",
    )

    plt.plot(
        this_country_test["date"],
        predictions_vector,
        label=f"Model Output",
        color="blue",
        linestyle="-",
        marker="",
    )
    # plot std
    if PLOT_STD:
        plt.fill_between(
            this_country_test["date"],
            predictions_vector - std_vector,
            predictions_vector + std_vector,
            color="blue",
            alpha=0.2,
        )

    # Adding title and labels
    plt.title(f"{country_name} Actual vs Predicted")
    plt.xlabel("Date")

    # turn dates 90 degrees
    plt.xticks(rotation=45)

    # make ticks more readable
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    # add vertical lines for the training and testing split
    plt.axvline(x=min_date_test, color="gray", linestyle="--", label="16 months gap")
    plt.axvline(x=max_date_train, color="gray", linestyle="--")

    plt.ylabel("Number of fatalities")
    plt.legend()

    # add light grid
    plt.grid(alpha=0.3)

    # Increment counters
    plots_added += 1

    if plots_added % num_plots_per_figure == 0 or index == len(country_ids) - 1:
        # Adjust layout and display the figure
        plt.tight_layout()

        if SHOW_PLOTS:
            plt.show()

        plt.figure(figsize=(15, 10))  # New figure
        plots_added = 0

if SHOW_PLOTS:
    plt.show()

if not SHOW_PLOTS:
    plt.close("all")

this_country_target

# +

if prediction_year == 2024:
    month_to_cut = 4
    test_df = test_df[test_df["date"] <= f"2024-{month_to_cut}-01"]
    test_df_new = test_df_new[test_df_new["date"] <= f"2024-{month_to_cut}-01"]

# +
# Assuming test_df is your DataFrame, and 'target' and 'predictions' are columns in it
unique_months = test_df["month_id"].unique()
print("Unique months:", unique_months)

# filter all point with actual more than 300
# test_df = test_df[test_df[target] < 300]

# Calculate the grid size for the subplot (simple square root approximation for a square grid)
n_unique_months: int = len(unique_months)

grid_size_x = int(n_unique_months**0.5) + (
    1 if n_unique_months % int(n_unique_months**0.5) else 0
)
grid_size_y = grid_size_x + 1

# print(f'Grid size: {grid_size}')

# Set overall figure size
plt.figure(figsize=(grid_size_x * 6, grid_size_y * 3))

for index, month_id in enumerate(unique_months, start=1):
    this_month = test_df[test_df["month_id"] == month_id]

    # mean_sq_error = sqrt(mean_squared_error(this_month[target], this_month['ngb_predictions']))
    current_date = this_month["date"].iloc[0]
    target_month = this_month[target]
    predictions_month = this_month["ngb_predictions"]

    # get this month max and min index
    month_start_index = this_month.index.min()
    month_end_index = this_month.index.max()
    mean_crps = calculate_CRPS(sampled_dist, y_test, month_start_index, month_end_index)

    # Create subplot for current month
    plt.subplot(grid_size_x, grid_size_y, index)
    plt.scatter(
        target_month,
        predictions_month,
        color="blue",
        label="Actual vs Predicted",
        alpha=0.5,
    )

    if PLOT_STD:
        predictions_std_month = this_month["ngb_predictions_std"]
        plt.errorbar(
            target_month,
            predictions_month,
            yerr=predictions_std_month,
            fmt="o",
            color="blue",
            alpha=0.5,
        )

    # print current_date in YY/MM format
    print_date = current_date.strftime("%Y-%m")

    plt.title(f"Month: {print_date}; mean CRPS: {mean_crps:.2f}", fontsize=11)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # plt.xscale('log')
    # plt.yscale('log')

    if not LOG_TRANSFORM:
        line_length = 2000  # default
    else:
        line_length = 8  # for log transformed

    plt.plot([0, line_length], [0, line_length], color="red", label="45 degree line")
    plt.legend()
    plt.xticks(rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()

# set font size to fit better
plt.rcParams.update({"font.size": 9 if LOG_TRANSFORM else 10})

if SAVE_FIGURES:
    plt.savefig(f"{figures_prefix}/NGBoost_predictions.png", dpi=300)

plt.close("all")
# -

if LOG_TRANSFORM:
    # Set overall figure size
    plt.figure(figsize=(grid_size_x * 6, grid_size_y * 3))

    for index, month_id in enumerate(unique_months, start=1):
        this_month = test_df[test_df["month_id"] == month_id]

        # mean_sq_error = sqrt(mean_squared_error(this_month[target], this_month['ngb_predictions']))
        current_date = this_month["date"].iloc[0]
        target_month = np.exp(this_month[target]) - 1
        predictions_month = np.exp(this_month["ngb_predictions"]) - 1

        # get this month max and min index
        month_start_index = this_month.index.min()
        month_end_index = this_month.index.max()
        mean_crps = calculate_CRPS(
            np.exp(sampled_dist) - 1,
            np.exp(y_test) - 1,
            month_start_index,
            month_end_index,
        )

        # Create subplot for current month
        plt.subplot(grid_size_x, grid_size_y, index)
        plt.scatter(
            target_month,
            predictions_month,
            color="blue",
            label="Actual vs Predicted",
            alpha=0.5,
        )

        if PLOT_STD:
            predictions_std_month = this_month["ngb_predictions_inverselog_std"]
            plt.errorbar(
                target_month,
                predictions_month,
                yerr=predictions_std_month,
                fmt="o",
                color="blue",
                alpha=0.5,
            )

        # print current_date in YY/MM format
        print_date = current_date.strftime("%Y-%m")

        plt.title(f"Month: {print_date}; mean CRPS: {mean_crps:.2f}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        # plt.xscale('log')
        # plt.yscale('log')

        line_length = 2000
        plt.plot(
            [0, line_length], [0, line_length], color="red", label="45 degree line"
        )
        plt.legend()
        plt.xticks(rotation=45)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # set font size to 18
    plt.rcParams.update({"font.size": 10})
    if SAVE_FIGURES:
        plt.savefig(f"{figures_prefix}/NGBoost_predictions_inverselog.png", dpi=300)

    plt.close("all")

if SHOW_PLOTS or SAVE_FIGURES:
    plt.close("all")
    shap.initjs()
    explainer = shap.TreeExplainer(
        ngb, model_output=0
    )  # use model_output = 1 for scale trees
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(
        shap_values, X_test, feature_names=X_test.columns, show=False, max_display=10
    )
    if SAVE_FIGURES:
        plt.savefig(f"{figures_prefix}/NGBoost_shap_values.png", dpi=300)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")

# +
## SHAP plot for loc trees
feature_importance_loc = ngb.feature_importances_[0]
feature_importance_loc = pd.DataFrame(
    feature_importance_loc, index=X_train.columns, columns=["importance"]
).sort_values(by="importance", ascending=False)

feature_importance_scale = ngb.feature_importances_[1]
feature_importance_scale = pd.DataFrame(
    feature_importance_scale, index=X_train.columns, columns=["importance"]
).sort_values(by="importance", ascending=False)

# save as list names of 35 least important features
# feature_importance_loc.tail(35).index.tolist()
feature_importance_loc

# +

# filter Ethiopia

# ethiopia_test = test_df[test_df['country_id'] == 57]
# ethiopia_test_max_error_index = test_df['ngb_predictions'].idxmax()
# X_test.iloc[ethiopia_test_max_error_index][[target]]
# X_test[ethiopia_test_max_error_index:ethiopia_test_max_error_index + 1]
# test_df['ngb_predictions'].idxmax()
# print country name
if SHOW_PLOTS:
    print(test_df.iloc[test_df["ngb_predictions"].idxmax()]["country_name"])
    shap.plots.force(
        explainer.expected_value,
        shap_values[test_df["ngb_predictions"].idxmax()],
        X_test.iloc[test_df["ngb_predictions"].idxmax()],
        matplotlib=True,
    )
    plt.show()

# +
# make a force plot for a separate prediction
countries_to_force_plot: list[str] = ["Ethiopia", "Turkey", "Algeria"]

for country_name in countries_to_force_plot:
    indexes_for_country = test_df[
        test_df["country_name"].str.contains(country_name)
    ].index

    for m, index in enumerate(indexes_for_country, start=1):
        shap.force_plot(
            explainer.expected_value,
            shap_values[index],
            X_test.iloc[index],
            matplotlib=True,
            show=False,
        )
        # add vectical of true value
        # plt.axvline(x=y_test.iloc[index], color='red', linestyle='--')
        plt.legend([f"Actual: {y_test.iloc[index]}"], loc="upper left")

        if SHOW_PLOTS:
            plt.show()

        if SAVE_FIGURES:
            plt.savefig(
                f"{figures_prefix}/force_plots/NGBoost_shap_fp_{country_name}_{m}_{index}.png",
                dpi=300,
            )

if not SHOW_PLOTS:
    plt.close("all")

# +
# https://stanfordmlgroup.github.io/ngboost/3-interpretation.html
# DO_IMPORTANCE = False
# # print all feature importance sorted
# feature_importance = bst.get_fscore()
# feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
# print(feature_importance)
#
# if DO_IMPORTANCE:
#     from xgboost import plot_importance
#
#     # plot
#     plot_importance(bst, max_num_features=10)
#     plt.show()
#
#     import shap
#
#     explainer = shap.TreeExplainer(bst)
#     # dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtrain.feature_names = X_train.columns.tolist()
#     explanation = explainer(dtrain)
#     explanation = shap.Explanation(
#         values=explanation.values,
#         base_values=explanation.base_values,
#         data=explanation.data,
#         feature_names=X_train.columns.tolist()
#     )
#     shap.plots.beeswarm(explanation)

# +
# len((np.exp(sampled_dist) - 1).std(axis=0))
# len(sampled_dist.std(axis=1))
#
# (np.exp(sampled_dist.std(axis=1)) - 1).min()

# +
# get observations with the highest error
# calculate the error
test_df["error"] = abs(test_df[target] - test_df["ngb_predictions"])

temp_df = test_df.sort_values(by="error", ascending=False)
highest_error_indices = temp_df.head(10).index
del temp_df
# highest_error_indices

# get the highest error
highest_error = test_df.nlargest(30, "error")

# drop columns that contain 'country_id_'
highest_error = highest_error[
    highest_error.columns.drop(list(highest_error.filter(regex="country_id_")))
]

# add country name
country_list = pd.read_csv("../data/country_list.csv")

highest_error = highest_error.merge(country_list, on="country_id")
highest_error
# from the test_df get sorted by highest, get the first 10
# -

# get mean error for Ethiopia
ethiopia_error = test_df[test_df["country_id"] == 57]
ethiopia_error["error"].mean()

# +
ngb_predictions_sampled = ngb_predictions_dist.sample(1000).T.astype(int)
#
# concat to test_df
# test_df = pd.concat([test_df, ngb_predictions_sampled], axis=1)


# negative_mask = ngb_predictions_sampled < 0
# # print how many negative values are there
# print(negative_mask.sum().sum()) # 917442
# # print total number of values
# print(negative_mask.size) # 2028000
# # print percentage of negative values
# print(negative_mask.sum().sum() / negative_mask.size) #0.452387573964497
# # sample once more and fill in the previous negative values with values from new distribution
# ngb_predictions_sampled[negative_mask] = ngb_predictions_dist.sample(1000).T[negative_mask]
# # print again how many negative values are there
# negative_mask =  ngb_predictions_sampled < 0
# print(negative_mask.sum().sum())  # 423012

# # set 0 if negative
# # ngb_predictions_sampled = ngb_predictions_sampled.clip(min=0)
# ngb_predictions_sampled = ngb_predictions_sampled
#
#
# # plot histogram of the sampled predictions using plt
plt.figure(figsize=(10, 6))
plt.hist(
    ngb_predictions_sampled[325], bins=50, alpha=0.7, label="NGB Predictions"
)  # MAX
# plot a dot for the actual value
# plt.scatter([actuals_model['ged_sb'].max()], [0], color='red', label='Actual Value')
# plt.hist(ngb_predictions_sampled[20], bins=50, alpha=0.7, label='NGB Predictions') # MIN
plt.title("Histogram of NGB Predictions")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.legend()

if SHOW_PLOTS:
    plt.show()

if not SHOW_PLOTS:
    plt.close("all")

# ngb_predictions_sampled[0]
# -

# len(set(benchmark_model['country_id'].unique()) - set(test_df['country_id'].unique()))
print(benchmark_model["month_id"].unique())
print(test_df["month_id"].unique())
print("Adjusted month_id for predictions:", test_df["month_id"].unique() + 15)
set(benchmark_model["month_id"].unique()) == set(test_df["month_id"].unique() + 15)

test_df.index

# +
# actuals_model.rename(columns={'ged_sb': 'actuals'}, inplace=True)
# actuals_model.set_index(['month_id', 'country_id'], inplace=True)
# test_df_new.set_index(['month_id', 'country_id'], inplace=True)

# +
# actuals_model.rename(columns={'ged_sb': 'actuals'}, inplace=True)
# test_df_new.reset_index(inplace=True, drop=True)
# actuals_model.reset_index(inplace=True, drop=True)


# drop actuals if it exists
# test_df_new = test_df_new.drop(columns='actuals', errors='ignore')

# join actuals to test_df_new
# test_df_new = test_df_new.join(actuals_model['actuals'], how='left')
# test_df_new.reset_index(inplace=True)
# actuals_model.reset_index(inplace=True)
# test_df_new
# actuals_model['actuals']
# test_df_new.head(10)
# test_df_new.head(1000)

# +
print(list(test_df_new.columns).index(target))
print(len(test_df_new.columns))
print(test_df_edge)

test_df_new.head(100)[["month_id", "country_id", target, "ngb_predictions"]]

# +
# test_df_edge = test_df_new.shape[1]
# test_df_new = pd.concat([test_df_new, pd.DataFrame(ngb_predictions_sampled)], axis=1)
# ngb_predictions_sampled[1] == test_df_new.iloc[1, test_df_edge:]

# +
# get id of test_df_new[target].max()
indices_to_plot = [test_df_new[target].idxmax(), test_df_new[target].idxmin()]

# add Pakistan 2020-07 or 2024-04 if prediction year is 2024
indices_to_plot.append(
    test_df_new[
        (test_df_new["country_name"] == "Pakistan")
        & (
            test_df_new["date"]
            == f"{prediction_year}-0{4 if prediction_year == 2024 else 7}"
        )
    ].index[0]
)
indices_to_plot.extend(highest_error_indices)
# -

test_df_new["date"].tail()

# +
for index_id in indices_to_plot:
    rowww = test_df_new.iloc[[index_id]]

    actual_pred = rowww[target].values[0]

    hist_data_temp = rowww.iloc[:, test_df_edge:].values[0]
    # keep only samples that are in 95% confidence interval

    # Plot histogram of the sampled predictions using plt
    plt.figure(figsize=(10, 6))
    plt.hist(hist_data_temp, bins=50, alpha=0.7, label="NGB Predictions")

    # Plot vertical lines for actual value, mean value, and other relevant predictions
    plt.axvline(
        x=actual_pred,
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="Actual Value",
    )
    plt.axvline(
        x=rowww["ngb_predictions"].values[0],
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label="Mean Value (NGB)",
    )
    # plt.axvline(
    #     x=rowww['predictions'].values[0],
    #     color='red',
    #     linestyle='dashed',
    #     linewidth=2,
    #     label='XGBoost Prediction'
    # )

    formatted_date = rowww["date"].dt.strftime("%Y-%m").values[0]
    country_name = rowww["country_name"].values[0]
    print(formatted_date)
    plt.title(
        f'Histogram of NGB Predictions and Actuals for {country_name} on {formatted_date} (month_id: {int(rowww["month_id"].values[0])})'
    )
    plt.xlabel("Predicted Value")
    plt.ylabel("Frequency")
    plt.legend()

    if SAVE_FIGURES:
        plt.savefig(
            f'{figures_prefix}/histograms/prediction_and_actuals_for_{country_name}_month_id_{int(rowww["month_id"].values[0])}.png',
            dpi=300,
        )

    if SHOW_PLOTS:
        plt.show()

if not SHOW_PLOTS:
    plt.close("all")

# -

test_df_new[test_df_new[target] == test_df_new[target].max()]["ngb_predictions"]

rowww[["month_id", "country_id", target, "ngb_predictions"]]

print(train_df.shape)
print(test_df.shape)

new_predictions[new_predictions["outcome"] < 0]

# +
# hist_color = 'blue'
mean_line_color = "blue"
actual_value_color = "black"

# set font size bigger
plt.rcParams.update({"font.size": 18})


# Define a function to plot histograms for a given instance
def plot_histograms(instance_to_plot):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    raw_predictions = test_df_new.iloc[instance_to_plot, test_df_edge:]
    processed_predictions = new_predictions.loc[
        (
            test_df_new.iloc[instance_to_plot]["month_id"] + prediction_window,
            test_df_new.iloc[instance_to_plot]["country_id"],
        )
    ][["outcome"]]
    metadata = test_df_new.iloc[instance_to_plot][
        ["month_id", "country_id", "country_name", "date", target]
    ]
    formatted_date = metadata["date"].strftime("%Y-%m")

    # Set title for the row of histograms
    # fig.suptitle(f'{metadata["country_name"]} on {formatted_date} (month_id: {int(metadata["month_id"])})', fontsize=16)

    # Raw predictions histogram
    raw_predictions = raw_predictions.apply(pd.to_numeric, errors="coerce")
    raw_prediction_crps = pscore(raw_predictions, metadata[target]).compute()[0]

    axs[0].hist(raw_predictions, bins=30, alpha=0.7)
    axs[0].axvline(
        raw_predictions.mean(), color=mean_line_color, linestyle="dashed", linewidth=2
    )
    axs[0].axvline(metadata[target], color=actual_value_color, linewidth=2)
    axs[0].set_title(f"Histogram of Raw Predictions\nCRPS: {raw_prediction_crps:.2f}")
    axs[0].set_xlabel("Predicted Values")
    axs[0].set_ylabel("Frequency")
    axs[0].legend(
        [f"Mean Value {raw_predictions.mean():.2f}", f"Actual Value {metadata[target]}"]
    )

    # Processed predictions histogram
    processed_predictions = processed_predictions.apply(pd.to_numeric, errors="coerce")
    processed_prediction_crps = pscore(
        processed_predictions["outcome"], metadata[target]
    ).compute()[0]

    axs[1].hist(processed_predictions, bins=30, alpha=0.7)
    axs[1].axvline(
        processed_predictions.mean().iloc[0],
        color=mean_line_color,
        linestyle="dashed",
        linewidth=2,
    )
    axs[1].axvline(metadata[target], color=actual_value_color)
    axs[1].set_title(
        f"Histogram of Resampled Predictions\nCRPS: {processed_prediction_crps:.2f}"
    )
    axs[1].set_xlabel("Predicted Values")
    axs[1].legend(
        [
            f"Mean Value {processed_predictions.mean().iloc[0]:.2f}",
            f"Actual Value {metadata[target]}",
        ]
    )

    # Raw predictions histogram with non-negative values
    raw_predictions_non_neg = raw_predictions.clip(lower=0)
    raw_prediction_non_neg_crps = pscore(
        raw_predictions_non_neg, metadata[target]
    ).compute()[0]
    axs[2].hist(raw_predictions_non_neg, bins=30, alpha=0.7)
    axs[2].axvline(
        raw_predictions_non_neg.mean(),
        color=mean_line_color,
        linestyle="dashed",
        linewidth=2,
    )
    axs[2].axvline(metadata[target], color=actual_value_color)
    axs[2].set_title(
        f"Histogram of Clipped Predictions\nCRPS: {raw_prediction_non_neg_crps:.2f}"
    )
    axs[2].set_xlabel("Predicted Values")
    axs[2].legend(
        [
            f"Mean Value {raw_predictions_non_neg.mean():.2f}",
            f"Actual Value {metadata[target]}",
        ]
    )

    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(
            f'{figures_prefix}/histograms/histograms_{metadata["country_name"]}_{formatted_date}.png'
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


# Plot histograms for the first set of indices
plot_histograms(indices_to_plot[8])

# Plot histograms for the second set of indices
plot_histograms(indices_to_plot[2])
