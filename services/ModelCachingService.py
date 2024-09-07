import os
import glob
import pickle
from datetime import datetime
from utilities.types import Model
from pathlib import Path


class ModelCachingService:
    def search_model_cache(self, directory: str) -> list[str] | None:
        model_files: list[str] = list(Path(directory).glob("model_*.p"))
        return model_files

    def cache_model(self, model: Model, directory: str) -> None:
        model_files: list[str] | None = self.search_model_cache(directory)

        if model_files is not None:
            for file_path in model_files:
                os.remove(file_path)

        time = datetime.now()
        model_path: str = (
            f"{directory}/"
            "model_"
            f"{time.day}_{time.month}_{time.year}_"
            f"{time.hour}_{time.minute:02}"
            ".p"
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    def get_model(self, model_path: str) -> Model:
        with open(model_path, "rb") as file:
            model: Model = pickle.load(file)

        return model


model_caching_service = ModelCachingService()
