import numpy as np
import pandas as pd
from CRPS import CRPS as pscore
from services import data_transformation_service


class ModelEvaluationService:
    def calculate_crps(
        self,
        sampled_dist: np.ndarray,
        y_true: pd.Series,
        index_start: int,
        index_end: int,
        inverse_log_transform: bool,
    ) -> np.floating:
        if inverse_log_transform:
            sampled_dist = data_transformation_service.inverse_log_transform(
                sampled_dist
            )
            y_true = data_transformation_service.inverse_log_transform(y_true)

        scores = []

        for indx, y_point in enumerate(y_true[index_start:index_end]):
            crps_score = pscore(sampled_dist[indx], y_point).compute()[0]
            scores.append(crps_score)

        return np.average(scores)

    def calculate_rmse(self, y_true: pd.Series, y_pred: pd.Series) -> np.floating:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


model_evaluation_service = ModelEvaluationService()
