from typing import Tuple
import numpy as np


def _sparsify_top_percentile(arr: np.ndarray, p: float) -> np.ndarray: ...

def sparse_assignment(cost_matrix: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]: ...

