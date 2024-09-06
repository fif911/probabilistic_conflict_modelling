import os
import warnings
from concurrent.futures import ProcessPoolExecutor, wait

import pandas as pd
import numpy as np
import optuna
import shap

from sklearn.tree import DecisionTreeRegressor
from ngboost.scores import CRPScore, LogScore
from ngboost.distns import Poisson, Normal, MultivariateNormal, Gamma
from ngboost import NGBRegressor

from model.least_important_features import (
    LEAST_IMPORTANT_FEATURES_V2_5,
    LEAST_IMPORTANT_FEATURES_V2_4,
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

# ---------------- Optuna ----------------
OPTUNA: bool = True

# ---------------- Config ----------------
PREDICTION_YEARS: list[int] = [2021, 2022, 2023]
CM_FEATURES_VERSION: str = "2.5"
PREDICTION_WINDOW: int = 14

SHOW_PLOTS: bool = False
PLOT_STD: bool = True
SAVE_FIGURES: bool = True
USE_CACHED_MODEL: bool = True
SAVE_PREDICTIONS: bool = True

LEAST_IMPORTANT_FEATURES: list[str] = (
    LEAST_IMPORTANT_FEATURES_V2_5
    if CM_FEATURES_VERSION == "2.5"
    else LEAST_IMPORTANT_FEATURES_V2_4
)
COLUMNS_TO_POP: list[str] = ["date", "country_id", "gw_statename", "month_id"]

default_run_config = {
    "drop_35_least_important": False,
    "include_country_id": True,
    "log_transform": True,
    "include_month_id": True,
    "drop_0_rows_percent": 20,
    "least_important_features": LEAST_IMPORTANT_FEATURES,
    "columns_to_pop": COLUMNS_TO_POP,
    "random_state": 42,
}
default_model_config = {
    # "col_sample": 1.0,
    "learning_rate": 0.01,
    "Dist": Normal,
    "Score": CRPScore,
    "n_estimators": 300,
    "minibatch_frac": 0.5,
    "early_stopping_rounds": None,
    "verbose": False,
}

default_base_learner_config = {
    "criterion": "friedman_mse",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": 5,
    "splitter": "best",
    "random_state": 42,
}
# ---------------- Paths ----------------
folder_to_str: str = (
    f"ng_boost_cm_v{CM_FEATURES_VERSION}_"
    f"pw_{PREDICTION_WINDOW}_"
    f"{default_model_config['Dist'].__name__.lower()}_"
    f"d_{default_run_config['drop_0_rows_percent']}_"
    f"n_{default_model_config['n_estimators']}_"
    f"s_{default_model_config['Score'].__name__.lower()}_"
    f"c_{str(default_run_config['include_country_id'])[0]}_"
    f"m_{str(default_run_config['include_month_id'])[0]}_"
    f"bsd_{default_base_learner_config['max_depth']}_"
    f"mbf_{default_model_config['minibatch_frac']}_"
    f"dli_{35 if default_run_config['drop_35_least_important'] else 0}_"
    f"log_{str(default_run_config['log_transform'])[0]}"
)

# ---------------- Utils ----------------
dist_map = {
    "Normal": Normal,
    "Poisson": Poisson,
    "MultivariateNormal": MultivariateNormal,
    "Gamma": Gamma,
}

score_map = {"CRPScore": CRPScore, "LogScore": LogScore}

# ---------------- Main ----------------
def sub_run(prediction_year, optuna, run_config, model_config, base_learner_config):
    # ---------------- Paths ----------------
    model_cache_prefix: str = (
        f"../model_cache/{folder_to_str}/window=Y{prediction_year}"
    )
    figures_prefix: str = f"../figures/{folder_to_str}/window=Y{prediction_year}"
    submission_prefix: str = f"../submission/{folder_to_str}"

    if not optuna:
        os.makedirs(f"{figures_prefix}/force_plots", exist_ok=True)
        os.makedirs(f"{figures_prefix}/histograms", exist_ok=True)
        os.makedirs(f"{figures_prefix}/line_plots", exist_ok=True)
        os.makedirs(f"{figures_prefix}/cut_comparison_histograms", exist_ok=True)
        os.makedirs(f"{model_cache_prefix}", exist_ok=True)

    # ---------------- Data ----------------
    cm_features, target = data_reading_service.read_features(
        prediction_year, CM_FEATURES_VERSION, PREDICTION_WINDOW, "../data"
    )
    if not optuna:
        benchmark_model = data_reading_service.read_benchmark(
            "poisson", prediction_year
        )
        actuals_model = data_reading_service.read_actuals(prediction_year)
        country_list = pd.read_csv("../data/country_list.csv")

        if run_config["log_transform"]:
            cm_features_original_target = cm_features[target]

    # ---------------- Data Preparation ----------------
    (
        train_df,
        test_df,
        train_df_popped_cols,
        test_df_popped_cols,
    ) = data_transformation_service.data_preprocess(cm_features, target, **run_config)

    X_train, y_train = data_splitting_service.x_y_split(train_df, target)
    X_test, y_test = data_splitting_service.x_y_split(test_df, target)

    # ---------------- Model Training ----------------
    if not optuna:
        model_files: list[str] | None = model_caching_service.search_model_cache(
            model_cache_prefix
        )
        cached_model_available = model_files is not None
        print(f"Cached model available: {cached_model_available}")

    if not optuna and USE_CACHED_MODEL and cached_model_available:
        ngb = model_caching_service.get_model(model_files[0])

    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            print(f"  - Training on {prediction_year}...")
            base_learner = DecisionTreeRegressor(**base_learner_config)

            config = {
                **model_config,
                "Base": base_learner,
            }

            ngb = model_training_service.train_model(
                X_train, y_train, None, None, NGBRegressor, config
            )

        print(f"  - Training on {prediction_year} done!")

        if not optuna:
            model_caching_service.cache_model(ngb, model_cache_prefix)

    # ---------------- Evaluation ----------------
    ngb_predictions_dist = ngb.pred_dist(X_test)
    sampled_dist = ngb_predictions_dist.sample(1000).T

    score = model_evaluation_service.calculate_crps(
        sampled_dist, y_test, 0, len(y_test), run_config["log_transform"]
    )

    print(f"  - Training on {prediction_year} done with score: {score}")

    if optuna:
        return score

    ngb_train_predictions = ngb.predict(X_train)
    ngb_predictions = ngb.predict(X_test)

    if not run_config["log_transform"]:
        ngb_train_predictions = [max(0, pred) for pred in ngb_train_predictions]
        ngb_predictions = [max(0, pred) for pred in ngb_predictions]

    # ---------------- Data ----------------
    test_df["ngb_predictions"] = ngb_predictions
    train_df["ngb_predictions"] = ngb_train_predictions

    # add date column back to test_df and add to each date shift of 14 months
    test_df["date"] = test_df_popped_cols["date"] + pd.DateOffset(
        months=PREDICTION_WINDOW
    )
    train_df["date"] = train_df_popped_cols["date"]

    test_df["country_id"] = test_df_popped_cols["country_id"]
    train_df["country_id"] = train_df_popped_cols["country_id"]

    test_df["country_name"] = test_df_popped_cols["gw_statename"]
    train_df["country_name"] = train_df_popped_cols["gw_statename"]

    if not run_config["include_month_id"]:
        test_df["month_id"] = test_df_popped_cols["month_id"]
        train_df["month_id"] = train_df_popped_cols["month_id"]

    test_df_new = pd.concat([test_df, pd.DataFrame(sampled_dist)], axis=1)

    # ---------------- Submission ----------------
    if PLOT_STD:
        ngb_predictions_std = sampled_dist.std(axis=1)
        test_df["ngb_predictions_std"] = ngb_predictions_std

    if run_config["log_transform"]:
        test_df[
            "ngb_predictions_inverselog_std"
        ] = data_transformation_service.inverse_log_transform(sampled_dist).std(axis=1)
        test_df[
            "ngb_predictions_inverselog"
        ] = data_transformation_service.inverse_log_transform(
            test_df["ngb_predictions"]
        )
        test_df["original_" + target] = cm_features_original_target

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
            prediction_window=PREDICTION_WINDOW,
            path=path,
            log_transform=run_config["log_transform"],
        )
    # ---------------- Plots ----------------
    if SHOW_PLOTS or SAVE_FIGURES:
        explainer = shap.TreeExplainer(
            ngb, model_output=0
        )  # use model_output = 1 for scale trees
        shap_values = explainer.shap_values(X_test)

        # -------- Shap summary plot
        plotting_service.shap_summary_plot(
            shap_values=shap_values,
            data=X_test,
            max_display=10,
            show_plots=SHOW_PLOTS,
            save_figures=SAVE_FIGURES,
            path=f"{figures_prefix}/NGBoost_shap_values.png",
        )

        # -------- Shap force plot
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

        # -------- Line plots
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
            prediction_window=PREDICTION_WINDOW,
            plot_std=PLOT_STD,
            cutoff_date=cutoff_date,
            log_transform=run_config["log_transform"],
            show_plots=SHOW_PLOTS,
            save_figures=SAVE_FIGURES,
            directory=f"{figures_prefix}/line_plots",
        )

        # -------- Prediction plots
        if prediction_year == 2024:
            month_to_cut = 4
            test_df = test_df[test_df["date"] <= f"2024-{month_to_cut}-01"]
            test_df_new = test_df_new[test_df_new["date"] <= f"2024-{month_to_cut}-01"]
        # Assuming test_df is your DataFrame, and 'target' and 'predictions' are columns in it
        unique_months = test_df["month_id"].unique()

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
        if run_config["log_transform"]:
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

        # -------- Prediction and actuals histograms
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
        highest_error = highest_error.merge(country_list, on="country_id")

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

        plotting_service.plot_prediction_and_actuals_histograms(
            data=test_df_new,
            target_column=target,
            indices_to_plot=indices_to_plot,
            show_plots=SHOW_PLOTS,
            save_figures=SAVE_FIGURES,
            directory=f"{figures_prefix}/histograms",
        )

        # -------- Cut comparison histograms
        if SAVE_PREDICTIONS:
            plotting_service.plot_cut_comparison(
                instance_to_plot=indices_to_plot[8],
                data=test_df_new,
                new_predictions=new_predictions,
                target_column=target,
                prediction_window=PREDICTION_WINDOW,
                show_plots=SHOW_PLOTS,
                save_figures=SAVE_FIGURES,
                directory=f"{figures_prefix}/cut_comparison_histograms",
            )
            plotting_service.plot_cut_comparison(
                instance_to_plot=indices_to_plot[2],
                data=test_df_new,
                new_predictions=new_predictions,
                target_column=target,
                prediction_window=PREDICTION_WINDOW,
                show_plots=SHOW_PLOTS,
                save_figures=SAVE_FIGURES,
                directory=f"{figures_prefix}/cut_comparison_histograms",
            )

    return score


def run(optuna, run_config, model_config, base_learner_config):
    print("Statring the run...")

    with ProcessPoolExecutor(len(PREDICTION_YEARS)) as exe:
        futures = [
            exe.submit(
                sub_run,
                prediction_year,
                optuna,
                run_config,
                model_config,
                base_learner_config,
            )
            for prediction_year in PREDICTION_YEARS
        ]
        wait(futures)

    results = [f.result() for f in futures]

    if optuna:
        return results


def objective(trial: optuna.Trial):
    optuna: bool = True

    run_config = {
        "drop_35_least_important": trial.suggest_categorical(
            "drop_35_least_important", [True, False]
        ),
        "include_country_id": trial.suggest_categorical(
            "include_country_id", [True, False]
        ),
        "log_transform": trial.suggest_categorical("log_transform", [True, False]),
        "include_month_id": trial.suggest_categorical(
            "include_month_id", [True, False]
        ),
        "drop_0_rows_percent": trial.suggest_int("drop_0_rows_percent", 0, 100),
        "least_important_features": LEAST_IMPORTANT_FEATURES,
        "columns_to_pop": COLUMNS_TO_POP,
        "random_state": None,
    }

    model_config = {
        # col_sample
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
        "Dist": dist_map[
            trial.suggest_categorical("dist", ["Normal"])
        ],  # Poisson, 'Gamma', 'MultivariateNormal' doesn't work
        "Score": score_map[
            trial.suggest_categorical("score", ["CRPScore"])
        ],  # "LogScore"
        "n_estimators": trial.suggest_int("n_estimators", 1, 5),
        "minibatch_frac": trial.suggest_float("minibatch_frac", 0.1, 1.0),
        "early_stopping_rounds": None,
        "verbose": False,
    }

    base_learner_config = {
        "criterion": trial.suggest_categorical(
            "bs_criterion", ["squared_error", "absolute_error", "friedman_mse"]
        ),
        "min_samples_split": trial.suggest_int("bs_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("bs_min_samples_leaf", 1, 20),
        "min_weight_fraction_leaf": trial.suggest_float(
            "bs_min_weight_fraction_leaf", 0.0, 0.5
        ),
        "max_depth": trial.suggest_int("bs_max_depth", 3, 10),
        "splitter": trial.suggest_categorical("bs_splitter", ["best", "random"]),
    }

    return run(optuna, run_config, model_config, base_learner_config)


if __name__ == "__main__":
    if OPTUNA:
        # Minimize every model's score
        study = optuna.create_study(directions=["minimize" for _ in range(len(PREDICTION_YEARS))])
        study.optimize(objective, n_jobs=1, n_trials=6)

        for trial in study.best_trials:
            print("-" * 64)
            print(dict(zip(PREDICTION_YEARS, trial.values)))
            print(trial.params)

        print("-" * 64)
    else:
        run(False, default_run_config, default_model_config, default_base_learner_config)
