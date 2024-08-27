from typing import Any, Optional, Callable
from typing_extensions import Protocol
from numpy.typing import ArrayLike
import numpy as np
import torch


class ManifoldLearner(Protocol):
    def fit_transform(self: 'ManifoldLearner',
                      X: np.ndarray) -> np.ndarray: pass


class ImageTransformer:

    _fe: ManifoldLearner
    _dm: Callable
    _pixels: tuple[int, int]
    _xrot: np.ndarray
    _coords: np.ndarray

    def __init__(self, feature_extractor: str | ManifoldLearner = 'tsne',
                 discretization: str = 'bin',
                 pixels: int | tuple[int, int] = (224, 224)) -> None: ...

    @staticmethod
    def _parse_pixels(pixels: int | tuple[int, int]) -> tuple[int, int]: ...

    @staticmethod
    def _parse_feature_extractor(
            feature_extractor: str | ManifoldLearner) -> ManifoldLearner: ...

    @classmethod
    def _parse_discretization(cls, method: str) -> Callable: ...

    @classmethod
    def coordinate_binning(cls, position: np.ndarray,
                           px_size: tuple[int, int]) -> np.ndarray: ...

    @staticmethod
    def lsap_optimal_solution(cost_matrix: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def lsap_heuristic_solution(cost_matrix: np.ndarray) -> np.ndarray: ...

    @classmethod
    def coordinate_optimal_assignment(cls, position: np.ndarray,
                                      px_size: tuple[int, int]
                                      ) -> np.ndarray: ...

    @classmethod
    def coordinate_heuristic_assignment(cls, position: np.ndarray,
                                        px_size: tuple[int, int]
                                        ) -> np.ndarray: ...

    @staticmethod
    def calculate_pixel_centroids(px_size: tuple[int, int]) -> np.ndarray: ...

    @staticmethod
    def clustered_cdist(positions: np.ndarray, centroids: np.ndarray,
                        k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def fit(self, X: np.ndarray, y: Optional[ArrayLike] = None,
            plot: bool = False) -> ImageTransformer: ...

    @property
    def pixels(self) -> tuple[int, int]: ...

    @pixels.setter
    def pixels(self, pixels: int | tuple[int, int]) -> None: ...

    @staticmethod
    def scale_coordinates(coords: np.ndarray, dim_max: ArrayLike
                          ) -> np.ndarray: ...

    def _calculate_coords(self) -> None: ...

    def transform(self, X: np.ndarray, img_format: str = 'rgb',
                  empty_value: int = 0) -> np.ndarray | torch.tensor: ...

    def fit_transform(self, X: np.ndarray, **kwargs: Any
                      ) -> np.ndarray | torch.tensor: ...

    def inverse_transform(self, img: np.ndarray) -> np.ndarray: ...

    def feature_density_matrix(self) -> np.ndarray: ...

    def coords(self) -> np.ndarray: ...

    @staticmethod
    def _minimum_bounding_rectangle(hull_points: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    def _mat_to_rgb(mat: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def _mat_to_pytorch(mat: np.ndarray) -> torch.tensor: ...


class MRepImageTransformer:

    _its: list[ImageTransformer]

    def __init__(self, feature_extractor: list[ManifoldLearner],
                 discretization: str | list[str] = 'bin',
                 pixels: int | tuple[int, int] = (224, 224)) -> None: ...

    def fit(self, X: np.ndarray, y: Optional[ArrayLike] = None,
            plot: bool = False) -> MRepImageTransformer: ...

    def transform(self, X: np.ndarray, img_format: str = 'rgb',
                  empty_value: int = 0, collate: str = 'sample',
                  return_index: bool = True
                  ) -> (np.ndarray | tuple[np.ndarray, np.ndarray]
                        | torch.tensor | tuple[torch.tensor, np.ndarray]): ...

    def fit_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray: ...

    @staticmethod
    def prediction_reduction(input: np.ndarray, index: np.ndarray,
                             reduction: str = "mean"
                             ) -> np.ndarray: ...