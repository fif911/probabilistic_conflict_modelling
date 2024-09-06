import pandas as pd


class DataSplittingService:
    def train_test_split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        last_month_id: int = data.loc[::, "month_id"].max()
        train_upper_bound: int = last_month_id - 14
        test_lower_bound: int = last_month_id - 11

        train: pd.DataFrame = data.loc[
            data["month_id"] <= train_upper_bound, ::
        ].reset_index(drop=True)
        test: pd.DataFrame = data.loc[
            data["month_id"] >= test_lower_bound, ::
        ].reset_index(drop=True)

        return train, test

    def x_y_split(
        self, data: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        X: pd.DataFrame = data.drop(target_column, axis=1)
        y: pd.Series = data.loc[::, target_column]

        return X, y


data_splitting_service = DataSplittingService()
