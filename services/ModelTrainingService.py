import pandas as pd
from utilities.types import Model


class ModelTrainingService:
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.DataFrame | None,
        model: Model,
        model_config: dict,
    ) -> Model:
        model_instance: Model = model(**model_config)
        model_instance.fit(X_train, y_train, X_val, y_val)

        return model_instance


model_training_service = ModelTrainingService()
