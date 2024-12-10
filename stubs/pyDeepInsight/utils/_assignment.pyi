import numpy as np


class AsymmetricGreedySearch:
    benefit: np.ndarray
    assignment: np.ndarray
    minimize: bool
    rows_idx: np.ndarray
    _iter: int
    column_mask: np.ndarray
    row_swap_idx: np.ndarray
    row_swap_delta: np.ndarray
    col_swap_idx: np.ndarray
    col_swap_delta: np.ndarray

    def __init__(self, benefit: np.ndarray, minimize: bool = False,
                 matrix_dtype: np.dtype = np.float32) -> None: ...

    @staticmethod
    def min_max_scale(a: np.ndarray) -> np.ndarray: ...

    def initialize(self, shuffle: bool = False) -> None: ...

    def _calc_row_swap_delta(self, row_idx: int) -> tuple[int, float]: ...

    def _update_row_swap_deltas(self) -> float: ...

    def _apply_row_swap(self, row_idx: int) -> None: ...

    def _calc_col_swap_delta(self, row_idx: int) -> tuple[int, float]: ...

    def _update_col_swap_deltas(self) -> float: ...

    def _apply_col_swap(self, row_idx: int) -> None: ...

    def calc_assignment_benefit(self) -> float: ...

    def optimize(self, shuffle: bool = False,
                 maximum_iterations: int | None = None) \
            -> tuple[np.ndarray, np.ndarray]: ...

def _sparsify_top_percentile(arr: np.ndarray, p: float) -> np.ndarray: ...

def sparse_assignment(cost_matrix: np.ndarray, p: float) -> np.ndarray: ...

