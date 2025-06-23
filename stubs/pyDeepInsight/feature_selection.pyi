from typing import Callable
from .image_transformer import ImageTransformer

import numpy as np
from torch import nn, Tensor


class CAMFeatureSelector:

    cam_method: str
    feature_coords: np.ndarray
    model: nn.Module
    target_layer: nn.Module

    def __init__(self, model: nn.Module,
                 it: ImageTransformer,
                 target_layer: str | nn.Module | None = None,
                 cam_method: str = "GradCAM") -> None: ...

    def _resolve_target_layer(self,
                              target_layer: str | nn.Module | None = None
                              ) -> nn.Module | None: ...

    def compute_cam(self, X: Tensor, y: Tensor,
                    batch_size: int = 1) -> np.ndarray: ...

    @staticmethod
    def flatten_cam(cams: np.ndarray, method: str = "mean") -> np.ndarray: ...

    @staticmethod
    def flatten_classes(labels: np.ndarray, cam: np.ndarray,
                        method: str = "mean") -> dict[int, np.ndarray]: ...

    def calculate_class_activations(self, X: Tensor, y: Tensor,
                                    batch_size: int = 1,
                                    flatten_method: str = "mean"
                                    ) -> dict[int, np.ndarray]: ...

    def select_class_features(self, cams: dict[int, np.ndarray],
                              threshold: float = 0.6
                              ) -> dict[int, np.ndarray]: ...

CAM_FUNCTIONS: dict[str, Callable]
