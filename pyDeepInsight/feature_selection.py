from typing import Union, Optional, Dict
from .image_transformer import ImageTransformer

import numpy as np
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader

import pytorch_grad_cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchcam.methods._utils import locate_candidate_layer


class CAMFeatureSelector:
    """Extract important features from a model using class activation mapping
    (CAM).

    This class takes a trained model and it's training data to identify
    important features to prediction of each class.
    """

    def __init__(self, model: nn.Module,
                 it: ImageTransformer,
                 target_layer: Union[str, nn.Module] = None,
                 cam_method: str = "GradCAM") -> None:
        """Generate a CAMFeatureSelector instance

        Args:
            model: trained CNN model
            it: ImageTransformer class used to create the images that trained
                the model
            target_layer: the target layer of the model, the name of the
                target layer, or "None" for the layer to be selected
                automatically
            cam_method: name of the CAM method from pytorch-grad-cam
        """

        if cam_method not in CAM_FUNCTIONS.keys():
            raise ValueError(
                f"Unknown cam_method {cam_method}. Valid methods are "
                f"{CAM_FUNCTIONS.keys()}"
            )
        self.cam_method = cam_method
        self.feature_coords = it.coords()
        self.model = model
        self.target_layer = self._resolve_target_layer(target_layer)

    def _resolve_target_layer(self,
                              target_layer: Union[str, nn.Module] = None
                              ) -> Optional[nn.Module]:
        """Convert layer name to layer or identify candidate layer if none
        given.

        Args:
            target_layer: the target layer of the model, the name of the
                target layer, or "None" for the layer to be selected
                automatically
        """
        resolved_target = None
        submodule_dict = dict(self.model.named_modules())
        if isinstance(target_layer, nn.Module):
            resolved_target = target_layer
        elif isinstance(target_layer, str):
            if target_layer not in submodule_dict.keys():
                raise ValueError(
                    f"Unable to find submodule {target_layer} in the model")
            else:
                resolved_target = submodule_dict[target_layer]
        elif target_layer is None:
            target_layer = locate_candidate_layer(self.model)
            resolved_target = submodule_dict[target_layer]

        return resolved_target

    def compute_cam(self, X: Tensor, y: Tensor,
                    batch_size: int = 1, use_cuda: bool = False,
                    ) -> np.ndarray:
        """Compute class activation map (CAM) for each image in X of classes y.

        Args:
            X: Tensor of input images
            y: Tensor of input labels
            batch_size: Batch size (default 1)
            use_cuda: Whether to use cuda for calculating CAMs
        Return:
            A numpy array of CAMs
        """
        activations = np.empty((0, X.shape[-2], X.shape[-1]))
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        func = CAM_FUNCTIONS[self.cam_method]
        self.model.eval()
        for i, data in enumerate(dl):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            cam = func(model=self.model, target_layers=[self.target_layer],
                       use_cuda=use_cuda)
            targets = [ClassifierOutputTarget(label) for label in labels]
            grayscale_cam = cam(input_tensor=inputs, targets=targets)
            activations = np.append(activations, grayscale_cam, axis=0)

        return activations

    @staticmethod
    def flatten_cam(cams: np.ndarray,
                    method: str = "mean") -> np.ndarray:
        """Flatten multiple CAMs into a single CAM.

        Args:
            cams: ndarray of CAMs
            method: Method to flatten CAMs. When used with feature selection
                threshold, max is essentially a union operation, and min is an
                intersect operation.
        Returns:
            A numpy array
        """
        if method == "mean":
            flat_cam = np.mean(cams, axis=0)
        elif method == "max":
            flat_cam = np.max(cams, axis=0)
        elif method == "min":
            flat_cam = np.min(cams, axis=0)
        else:
            raise ValueError(f"Unknown CAM flatten method {method}")
        return flat_cam

    @staticmethod
    def flatten_classes(labels: np.ndarray, cam: np.ndarray,
                        method: str = "mean") -> Dict[int, np.ndarray]:
        """

        Args:
            labels: class labels for cam array
            cam: ndarray of CAMs
            method: method to merge CAMs per class passed to .flatten_cam()
        Returns:
             A dictionary with class labels as keys and
             a single CAM as the value
        """
        cats = np.unique(labels)
        cat_cam = {}
        for cat in cats:
            cat_act = cam[np.where(labels == cat)[0], :, :]
            flat_act = CAMFeatureSelector.flatten_cam(
                cams=cat_act, method=method)
            cat_cam[cat] = flat_act
        return cat_cam

    def calculate_class_activations(self, X: Tensor, y: Tensor,
                                    batch_size: int = 1, flatten_method="mean"
                                    ) -> Dict[int, np.ndarray]:
        """Calculate CAM for each input then flatten for each class.

        Args:
            X: Tensor of input images
            y: Tensor of input labels
            batch_size: Batch size (default 1)
            flatten_method: Method to flatten CAMs for each class. 'max' is
                essentially a union operation, and 'min' is an
                intersect operation.
        Returns:
            A dictionary with classes as keys and the flattened CAM as values
        """
        use_cuda = X.is_cuda
        activations = self.compute_cam(X=X, y=y, batch_size=batch_size,
                                       use_cuda=use_cuda)
        y_cpu = y.detach().cpu().numpy()
        cat_cam = self.flatten_classes(y_cpu, activations,
                                       method=flatten_method)

        return cat_cam

    def select_class_features(self, cams: Dict[int, np.ndarray],
                              threshold: float = 0.6) -> Dict[int, np.ndarray]:
        """Select features for each class using class-specific CAMs. Input
        feature coordinates are filtered based on activation at same
        coordinates.

        Args:
            cams: A dictionary with classes as keys and a CAM as values
            threshold: Activation cutoff for feature importance
        Returns:
            A dictionary with classes as keys and an array of feature indices
            as values

        """
        class_idx = {}
        for cat, cam in cams.items():
            cam_pass = np.stack(np.where(cam >= threshold)).T
            it_pass = np.where(
                (self.feature_coords == cam_pass[:, None]).all(-1).any(-1)
            )[0]
            class_idx[cat] = it_pass
        return class_idx


CAM_FUNCTIONS = {
    "GradCAM": pytorch_grad_cam.GradCAM,
    "AblationCAM": pytorch_grad_cam.AblationCAM,
    "XGradCAM": pytorch_grad_cam.XGradCAM,
    "GradCAMPlusPlus": pytorch_grad_cam.GradCAMPlusPlus,
    "ScoreCAM": pytorch_grad_cam.ScoreCAM,
    "LayerCAM": pytorch_grad_cam.LayerCAM,
    "EigenCAM": pytorch_grad_cam.EigenCAM,
    "EigenGradCAM": pytorch_grad_cam.EigenGradCAM,
    "FullGrad": pytorch_grad_cam.FullGrad
}
