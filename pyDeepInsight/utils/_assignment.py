import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


class AsymmetricGreedySearch:
    """A Numpy implementation of the Asymmetric Greedy Search (AGS) algorithm
    for solving the linear sum assignment problem, as described in "A heuristic
    for the time constrained asymmetric linear sum assignment problem" by Brown
    et al. (2017) [DOI:10.1007/s10878-015-9979-2].

    This implementation efficiently optimizes row-to-column assignments to
    maximize the overall benefit (or minimize costs). The algorithm includes
    greedy initialization, iterative row and column swaps, and dynamic updates
    to swap benefit calculations.

    Attributes:
        benefit (numpy.ndarray): The scaled benefit (or cost) matrix
            for row-column assignments.
        minimize (bool): Minimize the benefit rather than maximize it.
        rows_idx (numpy.ndarray): Indices of rows in the assignment matrix.
        row_swap_idx (numpy.ndarray): Best row swap candidate for each row.
        row_swap_delta (numpy.ndarray): Change in benefit for the best row swap.
        col_swap_idx (numpy.ndarray): Best unassigned column swap for each row.
        col_swap_delta (numpy.ndarray): Change in benefit for the best column
            swap.
        assignment (numpy.ndarray): Current row-to-column assignment indices.
        column_mask (numpy.ndarray): Boolean mask indicating unassigned columns.
        _iter (int): Counter tracking the number of iterations performed during
            optimization.
    """

    def __init__(self, benefit, minimize=False, matrix_dtype=np.float32):
        """Initializes the AsymmetricGreedySearch algorithm.

        Args:
            benefit (numpy.ndarray): The benefit (or cost) matrix to optimize.
                Rows represent tasks and columns represent assignments.
            minimize (bool): If True, treats the `benefit` matrix as costs to
                minimize (default: False).
            matrix_dtype (numpy.dtype): Data type to use for storing the
                benefit matrix (default: np.float32).
        """
        self.benefit = self.min_max_scale(benefit).astype(matrix_dtype)
        self.minimize = minimize
        if self.minimize:
            self.benefit = 1 - self.benefit
        n_rows, n_cols = self.benefit.shape
        self.rows_idx = np.arange(n_rows)
        self.row_swap_idx = -1 * np.ones(n_rows, dtype=int)
        self.row_swap_delta = np.zeros(n_rows, dtype=float)
        self.col_swap_idx = -1 * np.ones(n_rows, dtype=int)
        self.col_swap_delta = np.zeros(n_rows, dtype=float)
        self.assignment = -1 * np.ones(n_rows, dtype=int)
        self.column_mask = np.ones(n_cols, dtype=bool)
        self._iter = 0

    @staticmethod
    def min_max_scale(a):
        """Scales an array to the range [0, 1] using min-max normalization.

        Args:
            a (numpy.ndarray): The array to scale.

        Returns:
            numpy.ndarray: The scaled array.
        """
        min_val = np.min(a)
        range_val = np.ptp(a)
        return (a - min_val) / range_val if range_val > 0 else np.zeros_like(a)

    def initialize(self, shuffle=False):
        """Initializes row-to-column assignments using a greedy approach.
        The `Initial(n,m)` function of the published algorithm.

        Args:
            shuffle (bool): If True, randomizes the order of row assignment
                (default: False).
        """
        n_rows, n_cols = self.benefit.shape
        assignment = np.empty(n_rows, dtype=np.int64)
        column_mask = np.ones(n_cols, dtype=bool)

        rows = self.rows_idx.copy()
        if shuffle:
            np.random.shuffle(rows)

        for row_idx in rows:
            available_bm = self.benefit[row_idx, column_mask]
            max_idx_in_avail = available_bm.argmax()
            max_idx = np.where(column_mask)[0][max_idx_in_avail]
            assignment[row_idx] = max_idx
            column_mask[max_idx] = False

        self.assignment = assignment
        self.column_mask = column_mask
        self.row_swap_idx = -1 * np.ones(n_rows, dtype=int)
        self.row_swap_delta = np.zeros(n_rows, dtype=float)
        self.col_swap_idx = -1 * np.ones(n_rows, dtype=int)
        self.col_swap_delta = np.zeros(n_rows, dtype=float)
        self._iter = 0

    def _calc_row_swap_delta(self, row_idx):
        """ Computes the change in benefit from swapping the column assignment
        of a given row with every other row and identifies the best swap.

        Args:
            row_idx (int): Index of the row to calculate the swap delta for.

        Returns:
            tuple: The index of the best swap row and the corresponding benefit
                delta.
        """
        swap_benefit = self.benefit[row_idx, self.assignment] + \
            self.benefit[:, self.assignment[row_idx]]
        curr_benefit = self.benefit[row_idx, self.assignment[row_idx]] + \
            self.benefit[self.rows_idx, self.assignment]
        benefit_delta = swap_benefit - curr_benefit
        benefit_delta[row_idx] = -1
        best_swap_idx = np.argmax(benefit_delta)
        best_swap_delta = benefit_delta[best_swap_idx]
        return best_swap_idx, best_swap_delta

    def _update_row_swap_deltas(self):
        """Updates the best row swap options and their corresponding benefit
        deltas for all rows. The `BestRowSwap(B,V)` function of the published
        algorithm.

        Returns:
            float: The maximum benefit delta among all row swaps.
        """
        best_swap_idx, best_swap_delta = np.stack(
            [self._calc_row_swap_delta(r) for r in self.rows_idx]).T
        self.row_swap_idx = best_swap_idx.astype(np.int64)
        self.row_swap_delta = best_swap_delta
        return np.amax(self.row_swap_delta)

    def _apply_row_swap(self, row_idx):
        """Applies the best row swap for the specified row and updates the swap
        benefit matrices accordingly. The `RowSwap(B,V,r)` function of the
        published algorithm.

        Args:
            row_idx (int): Index of the row to apply the swap for.
        """
        swap_idx = self.row_swap_idx[row_idx]
        # switch assignments
        self.assignment[[row_idx, swap_idx]] = \
            self.assignment[[swap_idx, row_idx]]
        affected_rows = np.where((self.row_swap_idx == row_idx) |
                                 (self.row_swap_idx == swap_idx))[0]
        # update row swap best benefits
        for idx in affected_rows:
            self.row_swap_idx[idx], self.row_swap_delta[
                idx] = self._calc_row_swap_delta(idx)
        # update the column swap best benefits
        for idx in [row_idx, swap_idx]:
            self.col_swap_idx[idx], self.col_swap_delta[
                idx] = self._calc_col_swap_delta(idx)

    def _calc_col_swap_delta(self, row_idx):
        """Computes the benefit of swapping the column assignment of a given row
        with the best available unassigned column.

        Args:
            row_idx (int): Index of the row to calculate the swap delta for.

        Returns:
            tuple: The index of the best unassigned column and the
            corresponding benefit delta.
        """
        available_benefits = self.benefit[row_idx, self.column_mask]
        if available_benefits.size == 0:
            return -1, 0

        best_avail_idx = available_benefits.argmax()
        best_swap_idx = np.flatnonzero(self.column_mask)[best_avail_idx]
        best_swap_delta = (
                self.benefit[row_idx, best_swap_idx] -
                self.benefit[row_idx, self.assignment[row_idx]]
        )
        return best_swap_idx, best_swap_delta

    def _update_col_swap_deltas(self):
        """Updates the best unassigned column swap options and their
        corresponding benefit deltas for all rows. The `BestColSwap(B,V)`
        function of the published algorithm.

        Returns:
            float: The maximum benefit delta among all column swaps.
        """
        num_rows, num_cols = self.benefit.shape
        if num_rows == num_cols:
            return -1 * np.ones(num_rows, dtype=int), np.zeros(num_rows,
                                                               dtype=float)
        available_columns = np.where(self.column_mask)[0]
        available_benefits = self.benefit[:, available_columns]
        best_indices = np.argmax(available_benefits, axis=1)
        best_swap_idx = available_columns[best_indices]
        best_swap_delta = available_benefits[self.rows_idx, best_indices] - \
                self.benefit[self.rows_idx, self.assignment]
        self.col_swap_idx = best_swap_idx
        self.col_swap_delta = best_swap_delta
        return np.amax(self.col_swap_delta)

    def _apply_col_swap(self, row_idx):
        """Applies the best column swap for the specified row and updates the
        swap benefit matrices accordingly. The `ColSwap(B,V,r)` function of the
        published algorithm

        Args:
            row_idx (int): Index of the row to apply the column swap for.
        """
        original_col = self.assignment[row_idx]
        new_col = self.col_swap_idx[row_idx]
        # update any rows with the new column as best
        affected_rows_mask = (self.row_swap_idx == original_col)
        if np.any(affected_rows_mask):
            affected_rows = np.where(affected_rows_mask)[0]
            new_swap_idx, new_swap_delta = np.stack(
                [self._calc_row_swap_delta(r) for r in affected_rows]).T
            self.row_swap_idx[affected_rows_mask] = new_swap_idx
            self.row_swap_delta[affected_rows_mask] = new_swap_delta
        # update assignment
        self.assignment[row_idx] = new_col
        # update any columns with best assigned to new column
        self.column_mask[original_col] = True
        self.column_mask[new_col] = False
        affected_cols_mask = (self.col_swap_idx == new_col)
        affected_cols = np.where(affected_cols_mask)[0]
        new_swap_idx, new_swap_delta = np.stack(
            [self._calc_col_swap_delta(r) for r in affected_cols]).T
        self.col_swap_idx[affected_cols_mask] = new_swap_idx
        self.col_swap_delta[affected_cols_mask] = new_swap_delta

    def calc_assignment_benefit(self):
        """Calculates the total assignment benefit based on the current
        row-to-column assignments.

        Returns:
            float: The total benefit of the current assignment.
        """
        benefit = self.benefit[self.rows_idx, self.assignment]
        if self.minimize:
            return (1 - benefit).sum()
        else:
            return benefit.sum()

    def optimize(self, shuffle=False, maximum_iterations=None):
        """A python implementation of the algorithm described in 'A heuristic
        for the time constrained asymmetric linear sum assignment problem'

        Args:
            shuffle (bool): If True, rows are shuffled before initialization
                (default: False).
            maximum_iterations (int, optional): Maximum number of iterations to
                run the optimization.

        Returns:
            tuple: A tuple of row indices and their assigned column indices.
        """
        self.initialize(shuffle=shuffle)
        brb_max = self._update_row_swap_deltas()
        bcb_max = self._update_col_swap_deltas()

        while max(brb_max, bcb_max) > 0:
            while max(brb_max, bcb_max) > 0:
                if brb_max > bcb_max:
                    r = np.argmax(self.row_swap_delta)
                    self._apply_row_swap(r)
                else:
                    r = np.argmax(self.col_swap_delta)
                    self._apply_col_swap(r)
                brb_max = np.amax(self.row_swap_delta)
                bcb_max = np.amax(self.col_swap_delta)
                self._iter += 1
                if maximum_iterations and self._iter > maximum_iterations:
                    return self.rows_idx, self.assignment
            brb_max = self._update_row_swap_deltas()
            bcb_max = self._update_col_swap_deltas()
        return self.rows_idx, self.assignment


def _sparsify_top_percentile(arr, p):
    """
    Convert a dense matrix to a sparse representation by retaining only the
    smallest top-percentile values in each row.

    Args:
        arr (ndarray): 2D array of shape (n_rows, n_cols) representing the
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
            cost_matrix (ndarray): 2D array of shape (n_rows, n_cols)
                representing the cost matrix.
            p (float): The fraction (0 < p <= 1) of the smallest values to
                retain per row during sparsification.

        Returns:
            tuple: Two arrays representing the row and column indices of the
                optimal assignment.
        """
    sparce_dist = _sparsify_top_percentile(cost_matrix, p)
    return min_weight_full_bipartite_matching(sparce_dist)
