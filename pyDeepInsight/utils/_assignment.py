import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def _sparsify_top_percentile(arr, p):
    """
    Convert a dense matrix to a sparse representation by retaining only the
    smallest top-percentile values in each row.

    Args:
        arr (np.ndarray): 2D array of shape (n_rows, n_cols) representing the
            input cost matrix.
        p (float): The fraction (0 < p <= 1) of the smallest values to retain
            per row.

    Returns:
        csr_matrix: A sparse matrix with only the top-percentile smallest
            values retained per row.
    """
    top_k = int(arr.shape[1] * p)
    for i in range(arr.shape[0]):
        row = arr[i]
        top_indices = np.argpartition(row, top_k)[:top_k]
        mask = np.ones_like(row, dtype=bool)
        mask[top_indices] = False
        row[mask] = 0
    return csr_matrix(arr)


def sparse_assignment(cost_matrix, p=0.1):
    """
        Perform sparse assignment using a cost matrix by retaining only the
        smallest top-percentile values in each row.

        Args:
            cost_matrix (np.ndarray): 2D array of shape (n_rows, n_cols)
                representing the cost matrix.
            p (float): The fraction (0 < p <= 1) of the smallest values to
                retain per row during sparsification.

        Returns:
            tuple: Two arrays representing the row and column indices of the
                optimal assignment.
        """
    sparce_dist = _sparsify_top_percentile(cost_matrix, p)
    return min_weight_full_bipartite_matching(sparce_dist)
