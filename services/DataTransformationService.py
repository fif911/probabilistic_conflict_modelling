import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.preprocessing import OneHotEncoder
from services.DataSplittingService import data_splitting_service


class DataTransformationService:
    def data_preprocess(
        self,
        cm_features: pd.DataFrame,
        target: str,
        columns_to_pop: list[str],
        least_important_features: list[str],
        drop_35_least_important: bool,
        include_country_id: bool,
        log_transform: bool,
        include_month_id: bool,
        drop_0_rows_percent: bool,
        random_state: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series],]:
        cm_features = cm_features.drop(
            columns=[
                "year",
                "ccode",
                "region",
                "region23",
                "country",
                "gleditsch_ward",
            ],
            errors="ignore",
        )

        if drop_35_least_important:
            cm_features = cm_features.drop(
                columns=least_important_features, errors="ignore"
            )

        if include_country_id:
            cm_features = self.include_country_id(cm_features)

        if log_transform:
            cm_features[target] = self.log_transform(cm_features.loc[::, target])

        train_df, test_df = data_splitting_service.train_test_split(cm_features)

        if drop_0_rows_percent > 0:
            train_df = self.drop_0_rows_percentage(
                train_df, target, drop_0_rows_percent, random_state=random_state
            )

        test_df_popped_cols: dict[
            str, pd.Series
        ] = data_transformation_service.pop_columns(test_df, columns_to_pop)
        train_df_popped_cols: dict[
            str, pd.Series
        ] = data_transformation_service.pop_columns(train_df, columns_to_pop)

        if include_month_id:
            test_df["month_id"] = test_df_popped_cols["month_id"]
            train_df["month_id"] = train_df_popped_cols["month_id"]

        return (
            train_df,
            test_df,
            train_df_popped_cols,
            test_df_popped_cols,
        )

    def include_country_id(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode the 'country_id' column in the input DataFrame and include the encoded features in the output DataFrame.
        """
        encoder: OneHotEncoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

        countries_encoded: np.ndarray = encoder.fit_transform(data[["country_id"]])

        # rename the columns
        df_countries_encoded: pd.DataFrame = pd.DataFrame(
            countries_encoded, columns=encoder.get_feature_names_out(["country_id"])
        ).drop(columns="country_id_1")

        # merge the encoded features with the original dataset
        new_data: pd.DataFrame = pd.concat([data, df_countries_encoded], axis=1)
        new_data = new_data.dropna()

        return new_data

    def drop_0_rows_percentage(
        self,
        data: pd.DataFrame,
        target_column: str,
        percentage: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """
        Randomly remove specified percentage of rows for which target_column equals 0
        """
        indices: pd.Series = data[data[target_column] == 0].index.to_series()
        num_to_drop: int = int(len(indices) * percentage / 100)
        indices_to_drop: list[int] = indices.sample(
            num_to_drop, random_state=random_state
        ).to_list()

        data = data.drop(indices_to_drop).reset_index(drop=True)

        return data

    def log_transform(self, x):
        return np.log(x + 1)

    def inverse_log_transform(self, x):
        return np.exp(x) - 1

    def pop_columns(
        self, data: pd.DataFrame, columns: list[str]
    ) -> dict[str, pd.Series]:
        """
        Pop specified columns from a pandas DataFrame and return them as a dictionary of Series.
        """
        popped_columns: list[pd.Series] = []

        for column in columns:
            popped_columns.append(data.pop(column))

        return {column: popped_columns[i] for i, column in enumerate(columns)}


data_transformation_service = DataTransformationService()
