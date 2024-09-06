from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class Model(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def pred_dist(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass
