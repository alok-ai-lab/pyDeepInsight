import numpy as np
from torch import nn, Tensor


def step_blur_kernel(kernel_size: int, amplification: float) -> np.ndarray: ...

def apply_blur_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray: ...

def step_blur(img: np.ndarray, kernel_size: int,
              amplification: float) -> np.ndarray: ...

class StepBlur2d(nn.Module):

    def_amp: float
    kernel: Tensor

    def __init__(self, kernel_size: Tensor, amplification: float): ...

    def forward(self, input: Tensor) -> Tensor: ...

    @staticmethod
    def step_kernel(kernel_size: int, amplification: float): ...

def imgaborfilt(image: np.ndarray, wavelength: float, orientation: float,
              SpatialFrequencyBandwidth: float,
              SpatialAspectRatio: float) -> np.ndarray: ...

class GaborFilter2d(nn.Module):

    def __init__(self, wavelength: float, orientation: float): ...

    def forward(self, img: Tensor) -> Tensor: ...

    @staticmethod
    def pil_to_tensor(img: np.ndarray) -> Tensor: ...

    @staticmethod
    def tensor_to_pil(img: Tensor) -> np.ndarray: ...
