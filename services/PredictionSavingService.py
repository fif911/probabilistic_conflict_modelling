import os
import numpy as np
import pandas as pd
from services import data_transformation_service


class PredictionSavingService:
    vector_length: int = 1000

    def save_predictions(
        self,
        df: pd.DataFrame,
        benchmark_model: pd.DataFrame,
        prediction_window: int,
        path: str,
        log_transform: bool = False,
    ) -> pd.DataFrame:
        """
        Save predictions to a parquet file

        Note that this method assumes that VECTOR_LENGTH of sampled predictions were added to each country's month in
        the input dataframe.
        """

        missing_countries: set[int] = set(benchmark_model["country_id"].unique()) - set(
            df["country_id"].unique()
        )
        all_countries: set[int] = set(df["country_id"].unique()).union(
            missing_countries
        )

        # for each month for each country create 20 draws of the prediction named outcome
        # the structure of the file should be month_id, country_id, draw, outcome
        predictions_list: list[dict] = []

        for month_id in df["month_id"].unique():
            for country_id in all_countries:
                this_country_month: pd.DataFrame = df.loc[
                    (df["month_id"] == month_id) & (df["country_id"] == country_id), ::
                ]

                if country_id in missing_countries:
                    outcomes = np.zeros(self.vector_length)
                else:
                    # Get last 1000 values of the country month raw
                    outcomes = this_country_month.iloc[
                        ::, -self.vector_length :
                    ].values[0]

                    if log_transform:
                        outcomes = data_transformation_service.inverse_log_transform(
                            outcomes
                        )
                        outcomes[outcomes < 0] = 0

                    # remove all values smaller than 0
                    non_negatives = outcomes[outcomes >= 0]
                    negative_counts: int = np.sum(outcomes < 0)

                    if negative_counts > 0:
                        if len(non_negatives) != 0:
                            # Sample from the non-negative distribution to replace negative values
                            # We assume the distribution of non-negatives is suitable for sampling
                            sampled_values = np.random.choice(
                                non_negatives, size=negative_counts
                            )
                            outcomes[outcomes < 0] = sampled_values
                        else:
                            outcomes[outcomes < 0] = 0

                    assert all(
                        outcomes >= 0
                    ), "There are still negative values in the outcomes"

                predictions_list.extend(
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
        predictions: pd.DataFrame = pd.DataFrame(predictions_list)

        predictions["month_id"] = predictions["month_id"].astype(int)
        predictions["country_id"] = predictions["country_id"].astype(int)
        predictions["draw"] = predictions["draw"].astype(int)
        predictions["outcome"] = predictions["outcome"].astype(int)

        # set index to month_id, country_id, draw
        predictions.set_index(["month_id", "country_id", "draw"], inplace=True)

        # create folder if it does not exist recursively
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            print("Ovewriting existing submission file")

        predictions.to_parquet(path)

        return predictions


prediction_saving_service: PredictionSavingService = PredictionSavingService()
