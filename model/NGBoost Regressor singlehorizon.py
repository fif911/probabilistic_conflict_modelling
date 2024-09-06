# # Imports

# +
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from ngboost.scores import CRPScore, LogScore
from ngboost.distns import Poisson, Normal, MultivariateNormal
from ngboost import NGBRegressor

from matplotlib import pyplot as plt
import shap

from model.least_important_features import (
    LEAST_IMPORTANT_FEATURES_V2_4,
    LEAST_IMPORTANT_FEATURES_V2_5,
)

from services import (
    data_reading_service,
    data_splitting_service,
    data_transformation_service,
    model_caching_service,
    model_evaluation_service,
    model_training_service,
    plotting_service,
    prediction_saving_service,
)

# -

# # Variables

# +
# Input data
prediction_year: int = 2022
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
LOG_TRANSFORM: bool = True

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
model_config = {
    "n_estimators": n_estimators,
    "verbose_eval": 10,
    "Dist": dist,
    "learning_rate": 0.01,
    "Score": score,
    "random_state": 42,
    "Base": base_learner,
    "minibatch_frac": minibatch_frac,
    # 'col_sample': 1.0,
    "early_stopping_rounds": None,
}
# -

LEAST_IMPORTANT_FEATURES: list[str] = (
    LEAST_IMPORTANT_FEATURES_V2_5
    if cm_features_version == "2.5"
    else LEAST_IMPORTANT_FEATURES_V2_4
)

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
# -

os.makedirs(f"{figures_prefix}/force_plots", exist_ok=True)
os.makedirs(f"{figures_prefix}/histograms", exist_ok=True)
os.makedirs(f"{figures_prefix}/line_plots", exist_ok=True)
os.makedirs(f"{figures_prefix}/cut_comparison_histograms", exist_ok=True)
os.makedirs(f"{model_cache_prefix}", exist_ok=True)

# # Data

cm_features, target = data_reading_service.read_features(
    prediction_year, cm_features_version, prediction_window, "../data"
)
cm_features.head(5)

# +
# load benchmark model
benchmark_model = data_reading_service.read_benchmark("poisson", prediction_year)

benchmark_model.head(5)
# -

# load actuals
actuals_model = data_reading_service.read_actuals(prediction_year)
actuals_model.head(5)

if SHOW_PLOTS:
    # plot target per month
    cm_features[target].plot()
    cm_features["ged_sb"].plot()
    cm_features["ged_sb_tlag_6"].plot()
    plt.legend()
    plt.show()

country_list = pd.read_csv("../data/country_list.csv")
country_list.head(5)

# # Data Preparation

# +
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
cm_features = cm_features.drop(
    columns=["year", "ccode", "region", "region23", "country", "gleditsch_ward"],
    errors="ignore",
)

if DROP_35_LEAST_IMPORTANT:
    cm_features = cm_features.drop(columns=LEAST_IMPORTANT_FEATURES, errors="ignore")

if INCLUDE_COUNTRY_ID:
    cm_features = data_transformation_service.include_country_id(cm_features)

if LOG_TRANSFORM:
    cm_features_original_target = cm_features[target]
    cm_features[target] = data_transformation_service.log_transform(cm_features[target])
# -

# # Train Test Split

train_df, test_df = data_splitting_service.train_test_split(cm_features)

if DROP_0_ROWS_PERCENT > 0:
    train_df = data_transformation_service.drop_0_rows_percentage(
        data=train_df,
        target_column=target,
        percentage=DROP_0_ROWS_PERCENT,
        random_state=42,
    )

# +
columns_to_pop: list[str] = ["date", "gw_statename", "country_id", "month_id"]

test_df_popped_cols = data_transformation_service.pop_columns(test_df, columns_to_pop)
train_df_popped_cols = data_transformation_service.pop_columns(train_df, columns_to_pop)

if INCLUDE_MONTH_ID:
    test_df["month_id"] = test_df_popped_cols["month_id"]
    train_df["month_id"] = train_df_popped_cols["month_id"]

# +
print(test_df_popped_cols["month_id"].unique())

print("Difference between benchmark and test month_id:")
print(benchmark_model["month_id"].min() - test_df_popped_cols["month_id"].max())
print(benchmark_model["month_id"].min() - test_df_popped_cols["month_id"].min())

train_df.head(5)
# -

# # X Y Split

X_train, y_train = data_splitting_service.x_y_split(train_df, target)
X_test, y_test = data_splitting_service.x_y_split(test_df, target)

# # Model

# +
# Model tuning:
# https://stanfordmlgroup.github.io/ngboost/2-tuning.html#Using-sklearn-Model-Selection
model_files: list[str] | None = model_caching_service.search_model_cache(
    model_cache_prefix
)
cached_model_available = model_files is not None
print(f"Cached model available: {cached_model_available}")

if USE_CACHED_MODEL and cached_model_available:
    ngb = model_caching_service.get_model(model_files[0])

else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ngb = model_training_service.train_model(
            X_train, y_train, X_test, y_test, NGBRegressor, model_config
        )

        print(f"Training NGB with {n_estimators} estimators and {score} score...")
    print("Done!")

    model_caching_service.cache_model(ngb, model_cache_prefix)
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

# average crps score
print(
    f"Average CRPS score: {model_evaluation_service.calculate_crps(sampled_dist, y_test, 0, len(y_test), LOG_TRANSFORM)}"
)

# +
# map predictions to months based on the test_df
test_df["ngb_predictions"] = ngb_predictions
train_df["ngb_predictions"] = ngb_train_predictions

# add date column back to test_df and add to each date shift of 14 months
test_df["date"] = test_df_popped_cols["date"] + pd.DateOffset(months=prediction_window)
train_df["date"] = train_df_popped_cols["date"]

test_df["country_id"] = test_df_popped_cols["country_id"]
train_df["country_id"] = train_df_popped_cols["country_id"]

test_df["country_name"] = test_df_popped_cols["gw_statename"]
train_df["country_name"] = train_df_popped_cols["gw_statename"]

if not INCLUDE_MONTH_ID:
    test_df["month_id"] = test_df_popped_cols["month_id"]
    train_df["month_id"] = train_df_popped_cols["month_id"]
# -

test_df_edge = test_df.shape[1]
test_df_new = pd.concat([test_df, pd.DataFrame(sampled_dist)], axis=1)

# +
# TODO: Improve metrics and use all metrics from the VIEWS competition
# Calculate RMSE
# train_rmse = sqrt(mean_squared_error(y_train, xg_lss_pred_train))
# actuals_rmse = sqrt(mean_squared_error(actuals_model['ged_sb'], predictions))
# benchmark_rmse = sqrt(mean_squared_error(y_test, benchmark_model['outcome']))
ngb_train_rmse = model_evaluation_service.calculate_rmse(y_train, ngb_train_predictions)
ngb_test_rmse = model_evaluation_service.calculate_rmse(y_test, ngb_predictions)
all_zeros_rmse = model_evaluation_service.calculate_rmse(y_test, [0] * len(y_test))

# added if cases because length need to be equal, and if year is not full,
# then we need to make benchmark_model['outcome'] and actuals_model['ged_sb']
if prediction_year == 2024:
    actuals_length = len(actuals_model["ged_sb"])
    actuals_bench_rmse = model_evaluation_service.calculate_rmse(
        actuals_model["ged_sb"], benchmark_model["outcome"][:actuals_length]
    )
else:
    actuals_bench_rmse = model_evaluation_service.calculate_rmse(
        actuals_model["ged_sb"], benchmark_model["outcome"]
    )

print("Cm features version:", cm_features_version)
print(f"Prediction year: {prediction_year}")
print(f"Include country_id: {INCLUDE_COUNTRY_ID}")
print(f"Include month_id: {INCLUDE_MONTH_ID}")
print(f"Drop train 0 rows: {DROP_0_ROWS_PERCENT}%")
print(f"Normal distribution: {dist == Normal}")
print(f"Number of estimators: {n_estimators}")
print(f"Score: {str(score)}")
print(f"Log transform: {LOG_TRANSFORM}")
print(f"Base learner max depth: {bs_max_depth}")
print(f"Minibatch fraction: {minibatch_frac}")

# print(f"XGB [train predictions] RMSE: {train_rmse}")
# print(f"XGB [test predictions]  RMSE YTEST VS PREDICTIONS: {rmse}")
print(f"\nNGB [train predictions] RMSE NGB: {ngb_train_rmse}")
print(f"NGB [test predictions]  RMSE NGB: {ngb_test_rmse}")

# print(f"RMSE YTEST VS ACTUALS: {actuals_rmse}')
# print(f"RMSE YTEST VS BENCHMARK: {benchmark_rmse}')
print(f"All Zeros: {all_zeros_rmse}")
print(f"\nBenchmark: RMSE ACTUALS VS BENCHMARK: {actuals_bench_rmse}")
# +
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

    # add std to test_df
    test_df["ngb_predictions_std"] = ngb_predictions_std

if LOG_TRANSFORM:
    test_df[
        "ngb_predictions_inverselog_std"
    ] = data_transformation_service.inverse_log_transform(sampled_dist).std(axis=1)
    test_df[
        "ngb_predictions_inverselog"
    ] = data_transformation_service.inverse_log_transform(test_df["ngb_predictions"])
    test_df["original_" + target] = cm_features_original_target
# -

if SAVE_PREDICTIONS:
    path: str = (
        f"{submission_prefix}/"
        "cm/"
        f"window=Y{prediction_year}/"
        f"{folder_to_str}_Y{prediction_year}"
        ".parquet"
    )

    new_predictions: pd.DataFrame = prediction_saving_service.save_predictions(
        df=test_df_new,
        benchmark_model=benchmark_model,
        prediction_window=prediction_window,
        path=path,
        log_transform=LOG_TRANSFORM,
    )

# # Plots Section

# ## Importance Plots

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
# -

if SHOW_PLOTS or SAVE_FIGURES:
    explainer = shap.TreeExplainer(
        ngb, model_output=0
    )  # use model_output = 1 for scale trees
    shap_values = explainer.shap_values(X_test)

    plotting_service.shap_summary_plot(
        shap_values=shap_values,
        data=X_test,
        max_display=10,
        show_plots=SHOW_PLOTS,
        save_figures=SAVE_FIGURES,
        path=f"{figures_prefix}/NGBoost_shap_values.png",
    )

    countries_to_force_plot: list[str] = ["Ethiopia", "Turkey", "Algeria"]
    plotting_service.shap_force_plot_countries(
        explainer=explainer,
        shap_values=shap_values,
        initial_data=test_df,
        X=X_test,
        y=y_test,
        countries=countries_to_force_plot,
        show_plots=SHOW_PLOTS,
        save_figures=SAVE_FIGURES,
        directory=f"{figures_prefix}/force_plots",
    )

# ## Line Plots

# +
cutoff_date = actuals_model["date"].iloc[-1].date().strftime("%Y-%m-%d")

if prediction_year == 2024:
    cutoff_date = pd.to_datetime(f"{cutoff_date}")

else:
    cutoff_date = pd.to_datetime(f"{prediction_year}-12-01")

plotting_service.plot_line_plots(
    country_list=country_list,
    train=train_df,
    test=test_df,
    target_column=target,
    prediction_window=prediction_window,
    plot_std=PLOT_STD,
    cutoff_date=cutoff_date,
    log_transform=LOG_TRANSFORM,
    show_plots=SHOW_PLOTS,
    save_figures=SAVE_FIGURES,
    directory=f"{figures_prefix}/line_plots",
)
# -

if prediction_year == 2024:
    month_to_cut = 4
    test_df = test_df[test_df["date"] <= f"2024-{month_to_cut}-01"]
    test_df_new = test_df_new[test_df_new["date"] <= f"2024-{month_to_cut}-01"]

# ## Predictions vs Actuals Plots

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

plotting_service.predictions_plot(
    data=test_df,
    y=y_test,
    sampled_dist=sampled_dist,
    target_column=target,
    unique_months=unique_months,
    grid_size_x=grid_size_x,
    grid_size_y=grid_size_y,
    plot_std=PLOT_STD,
    inverse_log_transform=False,
    save_figures=SAVE_FIGURES,
    path=f"{figures_prefix}/NGBoost_predictions.png",
)
if LOG_TRANSFORM:
    plotting_service.predictions_plot(
        data=test_df,
        y=y_test,
        sampled_dist=sampled_dist,
        target_column=target,
        unique_months=unique_months,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        plot_std=PLOT_STD,
        inverse_log_transform=True,
        save_figures=SAVE_FIGURES,
        path=f"{figures_prefix}/NGBoost_predictions_inverselog.png",
    )
# -

# ## Histograms Plots and Highest Error calculation

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
highest_error.head(10)
# from the test_df get sorted by highest, get the first 10
# -

# get mean error for Ethiopia
ethiopia_error = test_df[test_df["country_id"] == 57]
ethiopia_error["error"].mean()

# +
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

plotting_service.plot_prediction_and_actuals_histograms(
    data=test_df_new,
    target_column=target,
    indices_to_plot=indices_to_plot,
    show_plots=SHOW_PLOTS,
    save_figures=SAVE_FIGURES,
    directory=f"{figures_prefix}/histograms",
)
# -

# ## Predictions distribution

plotting_service.plot_predictions_distribution(sampled_dist, 325, SHOW_PLOTS)

# ## Сomparison of different type of handling negative values

# Because new_predictions datastructure is created in the predictions saving block
if SAVE_PREDICTIONS:
    plotting_service.plot_cut_comparison(
        instance_to_plot=indices_to_plot[8],
        data=test_df_new,
        new_predictions=new_predictions,
        target_column=target,
        prediction_window=prediction_window,
        show_plots=SHOW_PLOTS,
        save_figures=SAVE_FIGURES,
        directory=f"{figures_prefix}/cut_comparison_histograms",
    )
    plotting_service.plot_cut_comparison(
        instance_to_plot=indices_to_plot[2],
        data=test_df_new,
        new_predictions=new_predictions,
        target_column=target,
        prediction_window=prediction_window,
        show_plots=SHOW_PLOTS,
        save_figures=SAVE_FIGURES,
        directory=f"{figures_prefix}/cut_comparison_histograms",
    )
