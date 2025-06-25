from typing import Optional
import numpy as np


class Norm2Scaler:

    _min0: float
    _max: float

    def __init__(self) -> None: ...

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None): ...

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                      ) -> np.ndarray: ...

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                  ) -> np.ndarray: ...
