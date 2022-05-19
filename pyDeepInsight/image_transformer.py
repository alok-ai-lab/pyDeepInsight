import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import inspect

from typing import Union, Tuple, Any, Optional
from typing_extensions import Protocol
from numpy.typing import ArrayLike


class ManifoldLearner(Protocol):
    def fit_transform(self: 'ManifoldLearner',
                      X: np.ndarray) -> np.ndarray: pass


class ImageTransformer:
    """Transform features to an image matrix using dimensionality reduction

    This class takes in data normalized between 0 and 1 and converts it to a
    CNN compatible 'image' matrix

    """

    def __init__(self, feature_extractor: Union[str, ManifoldLearner] = 'tsne',
                 pixels: Union[int, Tuple[int, int]] = (224, 224),
                 random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None) -> None:
        """Generate an ImageTransformer instance

        Args:
            feature_extractor: string of value ('tsne', 'pca', 'kpca') or a
                class instance with method `fit_transform` that returns a
                2-dimensional array of extracted features.
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
            random_state: int or RandomState. Determines the random number
                generator, if present, of a string defined feature_extractor.
            n_jobs: The number of parallel jobs to run for a string defined
                feature_extractor.
        """
        self._fe = self._parse_feature_extractor(
            feature_extractor, random_state, n_jobs)
        self._pixels = self._parse_pixels(pixels)
        self._xrot = np.empty(0)
        self._coords = np.empty(0)

    @staticmethod
    def _parse_pixels(pixels: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Check and correct pixel parameter

        Args:
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        return pixels

    @staticmethod
    def _parse_feature_extractor(
            feature_extractor: Union[str, ManifoldLearner],
            random_state: Optional[int], n_jobs: Optional[int]
    ) -> ManifoldLearner:
        """Validate the feature extractor value passed to the
        constructor method and return correct method

        Args:
            feature_extractor: string of value ('tsne', 'pca', 'kpca') or a
                class instance with method `fit_transform` that returns a
                2-dimensional array of extracted features.
        """
        if isinstance(feature_extractor, str):
            fe = feature_extractor.casefold()
            if fe == 'tsne'.casefold():
                fe_func = TSNE(
                    n_components=2, metric='cosine',
                    random_state=random_state,
                    n_jobs=n_jobs)
            elif fe == 'pca'.casefold():
                fe_func = PCA(n_components=2,
                              random_state=random_state)
            elif fe == 'kpca'.casefold():
                fe_func = KernelPCA(
                    n_components=2, kernel='rbf',
                    random_state=random_state,
                    n_jobs=n_jobs)
            else:
                raise ValueError(
                    f"feature_extractor '{feature_extractor}' not valid")
        elif hasattr(feature_extractor, 'fit_transform') and \
                inspect.ismethod(feature_extractor.fit_transform):
            fe_func = feature_extractor
        else:
            raise TypeError('Parameter feature_extractor is not a '
                            'string nor has method "fit_transform"')
        return fe_func

    def fit(self, X: np.ndarray, y: ArrayLike = None,
            plot: bool = False) -> None:
        """Train the image transformer from the training set (X)

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            y: Ignored. Present for continuity with scikit-learn
            plot: boolean of whether to produce a scatter plot showing the
                feature reduction, hull points, and minimum bounding rectangle

        Returns:
            self: object
        """
        # perform dimensionality reduction
        x_new = self._fe.fit_transform(X.T)
        # get the convex hull for the points
        chvertices = ConvexHull(x_new).vertices
        hull_points = x_new[chvertices]
        # determine the minimum bounding rectangle
        mbr, mbr_rot = self._minimum_bounding_rectangle(hull_points)
        # rotate the matrix
        # save the rotated matrix in case user wants to change the pixel size
        self._xrot = np.dot(mbr_rot, x_new.T).T
        # determine feature coordinates based on pixel dimension
        self._calculate_coords()
        # plot rotation diagram if requested
        if plot is True:
            plt.scatter(x_new[:, 0], x_new[:, 1], s=1,
                        cmap=plt.cm.get_cmap("jet", 10), alpha=0.2)
            plt.fill(x_new[chvertices, 0], x_new[chvertices, 1],
                     edgecolor='r', fill=False)
            plt.fill(mbr[:, 0], mbr[:, 1], edgecolor='g', fill=False)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

    @property
    def pixels(self) -> Tuple[int, int]:
        """The image matrix dimensions

        Returns:
            tuple: the image matrix dimensions (height, width)

        """
        return self._pixels

    @pixels.setter
    def pixels(self, pixels: Union[int, Tuple[int, int]]):
        """Set the image matrix dimension

        Args:
            pixels: int or tuple with the dimensions (height, width)
            of the image matrix

        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        self._pixels = pixels
        # recalculate coordinates if already fit
        if hasattr(self, '_coords'):
            self._calculate_coords()

    def _calculate_coords(self) -> None:
        """Calculate the matrix coordinates of each feature based on the
        pixel dimensions.
        """
        ax0_coord = np.digitize(
            self._xrot[:, 0],
            bins=np.linspace(min(self._xrot[:, 0]), max(self._xrot[:, 0]),
                             self._pixels[0])
        ) - 1
        ax1_coord = np.digitize(
            self._xrot[:, 1],
            bins=np.linspace(min(self._xrot[:, 1]), max(self._xrot[:, 1]),
                             self._pixels[1])
        ) - 1
        self._coords = np.stack((ax0_coord, ax1_coord), axis=1)

    def transform(self, X: np.ndarray, img_format: str = 'rgb',
                  empty_value: int = 0) -> np.ndarray:
        """Transform the input matrix into image matrices

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
                where n_features matches the training set.
            img_format: The format of the image matrix to return.
                'scalar' returns an array of shape (M, N). 'rgb' returns
                a numpy.ndarray of shape (M, N, 3) that is compatible with PIL.
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0.

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """
        img_coords = pd.DataFrame(np.vstack((
            self._coords.T,
            X
        )).T).groupby([0, 1], as_index=False).mean()

        img_list = []
        blank_mat = np.zeros(self._pixels)
        if empty_value != 0:
            blank_mat[:] = empty_value
        for z in range(2, img_coords.shape[1]):
            img_matrix = blank_mat.copy()
            img_matrix[img_coords[0].astype(int),
                       img_coords[1].astype(int)] = img_coords[z]
            img_list.append(img_matrix)

        img_matrices = np.empty(0)
        if img_format == 'rgb':
            img_matrices = np.array([self._mat_to_rgb(m) for m in img_list])
        elif img_format == 'scalar':
            img_matrices = np.stack(img_list)
        else:
            raise ValueError(f"'{img_format}' not accepted for img_format")

        return img_matrices

    def fit_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Train the image transformer from the training set (X) and return
        the transformed data.

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """
        self.fit(X)
        return self.transform(X, **kwargs)

    def feature_density_matrix(self) -> np.ndarray:
        """Generate image matrix with feature counts per pixel

        Returns:
            img_matrix (ndarray): matrix with feature counts per pixel
        """
        fdmat = np.zeros(self._pixels)
        np.add.at(fdmat, tuple(self._coords.T), 1)
        return fdmat

    def coords(self) -> np.ndarray:
        """Get feature coordinates

        Returns:
            ndarray: the pixel coordinates for features
        """
        return self._coords.copy()

    @staticmethod
    def _minimum_bounding_rectangle(hull_points: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the smallest bounding rectangle for a set of points.

        Modified from JesseBuesking at https://stackoverflow.com/a/33619018
        Returns a set of points representing the corners of the bounding box.

        Args:
            hull_points : an nx2 matrix of hull coordinates

        Returns:
            (tuple): tuple containing
                coords (ndarray): coordinates of the corners of the rectangle
                rotmat (ndarray): rotation matrix to align edges of rectangle
                    to x and y
        """

        pi2 = np.pi / 2
        # calculate edge angles
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)
        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            -np.sin(angles),
            np.sin(angles),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))
        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)
        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)
        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)
        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        rotmat = rotations[best_idx]
        # generate coordinates
        coords = np.zeros((4, 2))
        coords[0] = np.dot([x1, y2], rotmat)
        coords[1] = np.dot([x2, y2], rotmat)
        coords[2] = np.dot([x2, y1], rotmat)
        coords[3] = np.dot([x1, y1], rotmat)

        return coords, rotmat

    @staticmethod
    def _mat_to_rgb(mat: np.ndarray) -> np.ndarray:
        """Convert image matrix to numpy rgb format

        Args:
            mat: {array-like} (M, N)

        Returns:
            An numpy.ndarray (M, N, 3) with original values repeated across
            RGB channels.
        """
        return np.repeat(mat[:, :, np.newaxis], 3, axis=2)


class LogScaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.
    """

    def __init__(self):
        self._min0 = None
        self._max = None
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self._min0 = X.min(axis=0)
        self._max = np.log(X + np.abs(self._min0) + 1).max()

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
