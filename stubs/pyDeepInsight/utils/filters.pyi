import numpy as np
import torch


def step_blur_kernel(kernel_size: int, amplification: float) -> np.ndarray: ...

def apply_blur_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray: ...

def step_blur(img: np.ndarray, kernel_size: int,
              amplification: float) -> np.ndarray: ...

class StepBlur2d(torch.nn.Module):

    def_amp: float
    kernel: torch.tensor

    def __init__(self, kernel_size: torch.tensor, amplification: float): ...

    def forward(self, input: torch.tensor) -> torch.tensor: ...

    @staticmethod
    def step_kernel(kernel_size: int, amplification: float): ...

def imgaborfilt(image: np.ndarray, wavelength: float, orientation: float,
              SpatialFrequencyBandwidth: float,
              SpatialAspectRatio: float) -> np.ndarray: ...

class GaborFilter2d(torch.nn.Module):

    def __init__(self, wavelength: float, orientation: float): ...

    def forward(self, img: torch.tensor) -> torch.tensor: ...

    @staticmethod
    def pil_to_tensor(img: np.ndarray) -> torch.tensor: ...

    @staticmethod
    def tensor_to_pil(img: torch.tensor) -> np.ndarray: ...
