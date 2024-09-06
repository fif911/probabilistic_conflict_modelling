import os
import glob
import pickle
from datetime import datetime
from utilities.types import Model


class ModelCachingService:
    def search_model_cache(self, directory: str) -> list[str] | None:
        model_files: list[str] = glob.glob(f"{directory}/model_*.p")

        return model_files if len(model_files) != 0 else None

    def cache_model(self, model: Model, directory: str) -> None:
        model_files: list[str] | None = self.search_model_cache(directory)

        if model_files is not None:
            for file_path in model_files:
                os.remove(file_path)

        time = datetime.now()
        model_path: str = (
            f"{directory}/"
            "model_"
            f"{time.day}.{time.month}.{time.year}_"
            f"{time.hour}:{time.minute:02}"
            ".p"
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    def get_model(self, model_path: str) -> Model:
        with open(model_path, "rb") as file:
            model: Model = pickle.load(file)

        return model


model_caching_service = ModelCachingService()
