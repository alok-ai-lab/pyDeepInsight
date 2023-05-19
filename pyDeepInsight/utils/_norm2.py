from typing import Optional
import numpy as np

class Norm2Scaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self._min0 = X.min(axis=0)
        self._max = np.log(X + np.abs(self._min0) + 1).max()
        return self

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        self._min0 = X.min(axis=0)
        X_norm = np.log(X + np.abs(self._min0) + 1)
        self._max = X_norm.max()
        return X_norm / self._max

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                  ) -> np.ndarray:
        X_norm = np.log(X + np.abs(self._min0) + 1).clip(0, None)
        return (X_norm / self._max).clip(0, 1)
