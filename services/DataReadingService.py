import pandas as pd
from utilities.views_utils import views_month_id_to_date


class DataReadingService:
    def read_features(
        self,
        prediction_year: int,
        cm_features_version: str,
        prediction_window: int,
        directory: str,
    ) -> tuple[pd.DataFrame, str]:
        cm_features: pd.DataFrame = pd.read_csv(
            f"{directory}/cm_features_v{cm_features_version}_Y{prediction_year}.csv"
        )

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

        cm_features["date"] = pd.to_datetime(cm_features["date"])
        cm_features["country_id"] = cm_features["country_id"].astype("category")

        return cm_features, target

    def read_benchmark(self, model_name: str, prediction_year: int) -> pd.DataFrame:
        model_names: dict[str, str] = {
            "bootstrap": "bm_cm_bootstrap_expanded_",
            "poisson": "bm_cm_last_historical_poisson_expanded_",
        }
        name: str = model_names[model_name]

        benchmark_model: pd.DataFrame = pd.read_parquet(
            f"../benchmarks/{name}{prediction_year}.parquet"
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
            benchmark_model.groupby(["month_id", "country_id"])
            .agg(agg_funcs)
            .reset_index()
        )

        # Flatten the multi-level columns resulting from aggregation
        benchmark_model.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in benchmark_model.columns.values
        ]

        # Rename columns
        benchmark_model.rename(
            columns={"outcome_mean": "outcome", "outcome_std": "outcome_std"},
            inplace=True,
        )

        if prediction_year == 2024:
            # SHIFT MONTHS BACK BY 7 MONTHS in benchmarks as it starts from.
            # TODO: Check if this is valid (probably not but this is a bug in inputs)
            # TODO: Report to PRIO
            # todo: add to service
            benchmark_model["month_id"] = benchmark_model["month_id"] - 6

        # add date column
        benchmark_model["date"] = views_month_id_to_date(
            benchmark_model.loc[::, "month_id"]
        )

        return benchmark_model

    def read_actuals(self, prediction_year: int) -> pd.DataFrame:
        actuals_model: pd.DataFrame = pd.read_parquet(
            f"../actuals/cm/window=Y{prediction_year}/cm_actuals_{prediction_year}.parquet"
        ).reset_index(drop=False)

        # actuals_model = actuals_model.groupby(['month_id', 'country_id']).mean().reset_index()
        actuals_model["date"] = views_month_id_to_date(
            actuals_model.loc[::, "month_id"]
        )
        print("Unique month_id:", actuals_model["month_id"].unique())

        # rename outcome to ged_sb
        actuals_model.rename(columns={"outcome": "ged_sb"}, inplace=True)

        return actuals_model


data_reading_service = DataReadingService()
