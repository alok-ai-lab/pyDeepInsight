import numpy as np
import torch
from functools import partial
from sklearn.preprocessing import quantile_transform
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import BisectingKMeans
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import inspect
import warnings

from .utils._assignment import sparse_assignment


class ImageTransformer:
    """Transforms normalized data into a CNN-compatible image. This class takes
    in data normalized between 0 and 1 and converts it to a CNN compatible
    'image' matrix. It utilizes dimensionality reduction techniques such as
    t-SNE, PCA, or Kernel PCA, followed by discretization methods to map the
    data onto a 2D image format.

    Attributes:
        _fe (ManifoldLearner): The feature extraction method used for
            dimensionality reduction. It must have a `fit_transform` method.
        _dm (Callable): The discretization method used to assign data points to
            pixel coordinates.
        _pixels (Tuple[int, int]): The dimensions of the image matrix (height, width) to
            which the data will be mapped.
        _xrot (np.ndarray): The rotated coordinates of the data after
            dimensionality reduction.
        _coords (np.ndarray): The final pixel coordinates assigned to the data
            points after discretization.
        DISCRETIZATION_OPTIONS (dict): A dictionary mapping discretization
            method names  to their corresponding class methods for pixel
            assignment.

    """

    DISCRETIZATION_OPTIONS = {
        'bin': 'coordinate_binning',
        'qtb': 'coordinate_quantile_transformation',
        'lsa': ('coordinate_assignment', 'lsa'),
        'ags': ('coordinate_assignment', 'ags'),
        'ags_batched': ('coordinate_assignment', 'ags_batched'),
        'sla': ('coordinate_assignment', 'sla')
    }

    def __init__(self, feature_extractor='tsne', discretization='bin',
                 pixels=(224, 224)):
        """Generate an ImageTransformer instance

        Args:
            feature_extractor (str or object): string of value ('tsne', 'pca',
                'kpca') or a class instance with method `fit_transform` that
                returns a 2-dimensional array of extracted features.
            discretization (str): Defines the method for discretizing
                dimensionally reduced data to pixel coordinates. Options are
                'bin', 'qtb', 'lsa', 'ags', and 'sla'.
            pixels (int or tuple): The size of the image matrix, either as a
                square integer or as a tuple (height, width).
        """
        self._fe = self._parse_feature_extractor(feature_extractor)
        self._dm = self._parse_discretization(discretization)
        self._pixels = self._parse_pixels(pixels)
        self._xrot = np.empty(0)
        self._coords = np.empty(0)

    @staticmethod
    def _parse_pixels(pixels):
        """Validates and formats the pixel size parameter.

        Args:
            pixels (int or tuple): The size of the image matrix, either as a
                square integer or as a tuple (height, width).
        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        return pixels

    @staticmethod
    def _parse_feature_extractor(feature_extractor):
        """Validates and returns the appropriate feature extraction method.

        Args:
            feature_extractor (str or object): string of value ('tsne', 'pca',
                'kpca') or a class instance with method `fit_transform` that
                returns a 2-dimensional array of extracted features.

        Returns:
            ManifoldLearner: The feature extraction class instance to be used.
        """
        if isinstance(feature_extractor, str):
            warnings.warn("Defining feature_extractor as a string of 'tsne'," +
                          " 'pca', or 'kpca' is depreciated. Please provide " +
                          " a class instance", DeprecationWarning)
            fe = feature_extractor.casefold()
            if fe == 'tsne'.casefold():
                fe_func = TSNE(n_components=2, metric='cosine')
            elif fe == 'pca'.casefold():
                fe_func = PCA(n_components=2)
            elif fe == 'kpca'.casefold():
                fe_func = KernelPCA(n_components=2, kernel='rbf')
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

    @classmethod
    def _parse_discretization(cls, method):
        """Validate the discretization value passed to the
        constructor method and return correct function

        Args:
            method (str): Options are 'bin', 'qtb', 'lsa', 'ags', and 'sla'.

        Returns:
            function: The discretization function corresponding to the method.
        """
        option = cls.DISCRETIZATION_OPTIONS.get(method)
        if option is None:
            raise ValueError(f"Unknown discretization method: {method}")

        if isinstance(option, str):
            return getattr(cls, option)
        elif isinstance(option, tuple):
            method_name, solver_name = option
            solver = cls._get_solver(solver_name)
            return partial(getattr(cls, method_name), solver=solver)
        else:
            raise TypeError(f"Invalid format for discretization option: {option}")

    @classmethod
    def coordinate_binning(cls, position, px_size):
        """Assigns pixel locations based on the binning of feature coordinates.

        Args:
            position (ndarray): A 2D array of feature coordinates.
            px_size (tuple): The dimensions of the image (height, width).

        Returns:
            ndarray: A 2D array of feature-to-pixel assignments.
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_binned = np.floor(scaled).astype(int)
        # Need to move maximum values into the lower bin
        px_binned[:, 0][px_binned[:, 0] == px_size[0]] = px_size[0] - 1
        px_binned[:, 1][px_binned[:, 1] == px_size[1]] = px_size[1] - 1
        return px_binned

    @classmethod
    def coordinate_quantile_transformation(cls, position, px_size):
        """Assigns pixel locations based on quantile-transformed feature
        coordinates.

        Args:
            position (ndarray): A 2D array of feature coordinates.
            px_size (tuple): The dimensions of the image (height, width).

        Returns:
            ndarray: A 2D array of feature-to-pixel assignments.
        """
        trans_height = quantile_transform(position[:, 0, None],
                                          n_quantiles=px_size[0],
                                          output_distribution='uniform'
                                          ).flatten()
        trans_width = quantile_transform(position[:, 1, None],
                                         n_quantiles=px_size[1],
                                         output_distribution='uniform'
                                         ).flatten()
        trans_position = np.stack((trans_height, trans_width), axis=1)
        scaled = cls.scale_coordinates(trans_position, px_size)
        px_binned = np.floor(scaled).astype(int)
        # Need to move maximum values into the lower bin
        px_binned[:, 0] = np.clip(px_binned[:, 0], 0, px_size[0] - 1)
        px_binned[:, 1] = np.clip(px_binned[:, 1], 0, px_size[1] - 1)
        return px_binned

    @staticmethod
    def _get_solver(name):
        if name == 'lsa':
            solver = linear_sum_assignment
        elif name == 'sla':
            solver = partial(sparse_assignment, p=1/3)
        elif name == 'ags':
            from asymmetric_greedy_search import AsymmetricGreedySearch
            ags = AsymmetricGreedySearch(backend="numba")
            solver = partial(ags.optimize, minimize=True, shuffle=True)
        elif name == 'ags_batched':
            from asymmetric_greedy_search import AsymmetricGreedySearch
            ags = AsymmetricGreedySearch(backend="numba")
            solver = partial(ags.optimize, minimize=True, shuffle=True, row_batch_size=5)
        else:
            raise ValueError(f"Unknown solver {name}")
        return solver

    @classmethod
    def assignment_preprocessing(cls, position, px_size, max_assignments):
        """Preprocesses features by clustering and calculating the square of
        the Euclidean distances to the pixel centers.

        Args:
            position (ndarray): A 2D array of feature coordinates.
            px_size (tuple): The dimensions of the image (height, width).
            max_assignments (int): The maximum number of clusters to generate.

        Returns:
            tuple: A tuple of the distance matrix and the feature-to-cluster
                mappings.
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_centers = cls.calculate_pixel_centroids(px_size)
        # calculate distances
        if scaled.shape[0] > max_assignments:
            dist, labels = cls.clustered_cdist(scaled, px_centers,
                                               max_assignments)
        else:
            dist = cdist(scaled, px_centers, metric='euclidean')
            labels = np.arange(scaled.shape[0])
        dist = dist**2
        return dist, labels

    @classmethod
    def assignment_postprocessing(cls, position, px_size, solution, labels):
        """Assigns pixel coordinates to features based on the optimization
        solution.

        Args:
            position (ndarray): A 2D array of feature coordinates.
            px_size (tuple): The dimensions of the image (height, width).
            solution (ndarray): The assignment solution.
            labels (ndarray): The feature-to-assignment cluster mappings.

        Returns:
            ndarray: An array of feature-to-pixel assignments.
        """
        px_assigned = np.empty(position.shape, dtype=int)
        for i in range(position.shape[0]):
            # The feature at i
            # Is mapped to the cluster j=labels[i]
            # Which is mapped to the pixel center px_centers[j]
            # Which is mapped to the pixel k = lsa[1][j]
            # For pixel k, x = k % px_size[0] and y = k // px_size[0]
            j = labels[i]
            ki = solution[j].item()
            yi, xi = divmod(ki, px_size[0])
            px_assigned[i] = [yi, xi]
        return px_assigned

    @classmethod
    def coordinate_assignment(cls, position, px_size, solver):
        """Assigns pixel locations using an assignment solution on
        the distance matrix.

        Args:
            position (ndarray): A 2D array of feature coordinates.
            px_size (tuple): The dimensions of the image (height, width).
            solver (callable): The assignment solution function.

        Returns:
            ndarray: A 2D array of feature-to-pixel assignments.
        """
        # calculate distances
        k = int(np.prod(px_size))
        dist, labels = cls.assignment_preprocessing(position, px_size, k)
        # assignment of features/clusters to pixels
        assignment = solver(dist)[1]
        px_assigned = cls.assignment_postprocessing(position, px_size,
                                                    assignment, labels)
        return px_assigned

    @staticmethod
    def calculate_pixel_centroids(px_size):
        """Generate a 2d array of the centroid of each pixel

        Args:
            px_size (tuple): Image dimensions (height, width).

        Returns:
            ndarray: A 2D array of pixel centroid locations with shape
                (height * width, 2).
        """
        px_map = np.empty((np.prod(px_size), 2))
        for i in range(0, px_size[0]):
            for j in range(0, px_size[1]):
                px_map[i * px_size[0] + j] = [i, j]
        px_centroid = px_map + 0.5
        return px_centroid

    @staticmethod
    def clustered_cdist(positions, centroids, k):
        """Cluster the features into k clusters and then calculate the distance
        from the clusters to the (pixel) centroids.

        Args:
            positions (ndarray): Location of the features
                (n_samples, n_features).
            centroids (ndarray): The center of the pixels (n_pixels, 2).
            k (int): The number of clusters to generate.

        Returns:
            tuple: A tuple containing:
                - dist (ndarray): Distance (cost) matrix of shape (k, n_pixels).
                - cl_labels (ndarray): Array of shape (n_samples,) with cluster
                labels.
        """
        kmeans = BisectingKMeans(n_clusters=k).fit(positions)
        cl_labels = kmeans.labels_
        cl_centers = kmeans.cluster_centers_
        dist = cdist(cl_centers, centroids, metric='euclidean')
        return dist, cl_labels

    def fit(self, X, y=None, plot=False):
        """Train the image transformer from the training set (X)

        Args:
            X (array-like): The training data of shape (n_samples, n_features)
            y: Ignored. Present for continuity with scikit-learn.
            plot (bool): Whether to produce a scatter plot showing the feature
                reduction, hull points, and minimum bounding rectangle.

        Returns:
            self: The fitted instance of the image transformer.
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
            plt.scatter(x_new[:, 0], x_new[:, 1], s=1, alpha=0.2)
            plt.fill(x_new[chvertices, 0], x_new[chvertices, 1],
                     edgecolor='r', fill=False)
            plt.fill(mbr[:, 0], mbr[:, 1], edgecolor='g', fill=False)
        return self

    @property
    def pixels(self):
        """The image matrix dimensions

        Returns:
            tuple: The image matrix dimensions (height, width)

        """
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        """Set the image matrix dimension

        Args:
            pixels (int or tuple): The dimensions (height, width) of the image
                matrix. If an integer is provided, it is treated as a
                square (height, width).

        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        self._pixels = pixels
        # recalculate coordinates if already fit
        if hasattr(self, '_coords'):
            self._calculate_coords()

    @staticmethod
    def scale_coordinates(coords, dim_max):
        """Scale a list of n-dimensional coordinates between zero and the
        given dimensional maximum.

        Args:
            coords (ndarray): A 2D ndarray of coordinates
                (n_samples, n_features).
            dim_max (list): Maximum range values for each dimension of `coords`.

        Returns:
            ndarray: A 2D ndarray of scaled coordinates (n_samples, n_features).
        """
        data_min = coords.min(axis=0)
        data_max = coords.max(axis=0)
        std = (coords - data_min) / (data_max - data_min)
        scaled = np.multiply(std, dim_max)
        return scaled

    def _calculate_coords(self):
        """Calculate the matrix coordinates of each feature based on the
        pixel dimensions.
        """
        px_coords = self._dm(self._xrot, self._pixels)
        self._coords = px_coords

    def transform(self, X, img_format='rgb', empty_value=0):
        """Transform the input matrix into image matrices.

        Args:
            X (array-like): Input matrix of shape (n_samples, n_features) where
                n_features matches the training set.
            img_format (str): The format of the image matrix to return.
                - 'scalar': returns a matrix of shape (n_samples, H, W).
                - 'rgb': returns a ndarray of shape (n_samples, H, W, 3).
                - 'pytorch': returns a tensor of shape (n_samples, 3, H, W).
            empty_value (int or float): Numeric value to fill elements where no
                features are mapped. Default is 0.

        Returns:
            ndarray or Tensor: A list of image matrices with dimensions
                defined by `pixels`.
        """
        unq, idx, cnt = np.unique(self._coords, return_inverse=True,
                                  return_counts=True, axis=0)
        img_matrix = np.zeros((X.shape[0],) + self._pixels)
        if empty_value != 0:
            img_matrix[:] = empty_value
        for i, c in enumerate(unq):
            img_matrix[:, c[0], c[1]] = X[:, np.where(idx == i)[0]].mean(axis=1)

        if img_format == 'rgb':
            img_matrix = self._mat_to_rgb(img_matrix)
        elif img_format == 'scalar':
            pass
        elif img_format == 'pytorch':
            img_matrix = self._mat_to_pytorch(img_matrix)
        else:
            raise ValueError(f"'{img_format}' not accepted for img_format")
        return img_matrix

    def fit_transform(self, X, **kwargs):
        """Train the image transformer from the training set (X) and return
        the transformed data.

        Args:
            X (array-like): The training data of shape (n_samples, n_features).

        Returns:
            ndarray or Tensor: A list of image matrices with dimensions
                defined by `pixels`.
        """
        self.fit(X)
        return self.transform(X, **kwargs)

    def inverse_transform(self, img):
        """Transform an image layer back to its original space.
            Args:
                img (ndarray): Image matrix of shape (B*, H, W, C*) where B and
                    C are optional.

            Returns:
                ndarray: The transformed feature matrix of shape
                    (n_samples, n_features).
        """
        if img.ndim == 2 and img.shape == self._pixels:
            X = img[self._coords[:, 0], self._coords[:, 1]]
        elif img.ndim == 3 and img.shape[-2:] == self._pixels:
            X = img[:, self._coords[:, 0], self._coords[:, 1]]
        elif img.ndim == 3 and img.shape[0:2] == self._pixels:
            X = img[self._coords[:, 0], self._coords[:, 1], :]
        elif img.ndim == 4 and img.shape[1:3] == self._pixels:
            X = img[:, self._coords[:, 0], self._coords[:, 1], :]
        else:
            raise ValueError((f"Expected dimensions of (B, {self._pixels[0]}, "
                              f"{self._pixels[1]}, C) where B and C are "
                              f"optional, but got {img.shape}"))
        return X

    def feature_density_matrix(self):
        """Generate image matrix with feature counts per pixel.

        Returns:
            ndarray: Matrix with feature counts per pixel of shape (H, W).
        """
        fdmat = np.zeros(self._pixels)
        np.add.at(fdmat, tuple(self._coords.T), 1)
        return fdmat

    def coords(self):
        """Get feature coordinates.

        Returns:
            ndarray: Pixel coordinates for features.
        """
        return self._coords.copy()

    @staticmethod
    def _minimum_bounding_rectangle(hull_points):
        """Find the smallest bounding rectangle for a set of points. Modified
        from JesseBuesking at https://stackoverflow.com/a/33619018.

        Args:
            hull_points (ndarray): nx2 matrix of hull coordinates.

        Returns:
            tuple: A tuple containing:
                - coords (ndarray): Coordinates of the corners of the rectangle.
                - rotmat (ndarray): Rotation matrix to align edges of rectangle.
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
    def _mat_to_rgb(mat):
        """Convert image matrix to numpy RGB format.

        Args:
            mat (ndarray): Matrix of shape (..., M, N).

        Returns:
            ndarray: RGB formatted matrix of shape (..., M, N, 3).
        """

        return np.repeat(mat[..., np.newaxis], 3, axis=-1)

    @staticmethod
    def _mat_to_pytorch(mat):
        """Convert image matrix to PyTorch tensor format.

        Args:
            mat (ndarray): Matrix of shape (..., M, N).

        Returns:
            Tensor: Tensor of shape (..., 3, M, N).
        """

        return torch.from_numpy(mat).unsqueeze(1).repeat(1, 3, 1, 1)


class MRepImageTransformer:
    """Transform features to multiple image matrices using dimensionality
    reduction. This class takes in data normalized between 0 and 1 and converts
    it to CNN compatible 'image' matrices

    """

    def __init__(self, feature_extractor, discretization='bin',
                 pixels=(224, 224)):
        """Initialize an MRepImageTransformer instance.

        Args:
            feature_extractor (list): A list of dimensionality reduction class
                instances with a `fit_transform` method returning a 2D array
                of extracted features. Alternatively, a list of tuples where
                the first element is the class instance and the second is
                a discretization option.
            discretization (str): specifying the default method for
                discretizing dimensionally reduced data into pixel coordinates
                if not provided in `feature_extractor`.
            pixels (int or tuple): The size of the image matrix, either as a
                square integer or as a tuple (height, width).
        """
        self.discretization = discretization
        self._its = []
        self.pixels = pixels
        self._data = None
        for it_cfg in feature_extractor:
            it = self.initialize_image_transformer(it_cfg)
            self._its.append(it)

    def initialize_image_transformer(self, config):
        """Create an ImageTransformer instance

        Args:
            config: A dimensionality reduction class instance or a tuple
                containing the instance and a discretization method.

        Returns:
            An initialized ImageTransformer instance.
        """
        if isinstance(config, (tuple, list)):
            return ImageTransformer(feature_extractor=config[0],
                                    discretization=config[1],
                                    pixels=self.pixels)
        else:
            return ImageTransformer(feature_extractor=config,
                                    discretization=self.discretization,
                                    pixels=self.pixels)

    def fit(self, X, y=None, plot=False):
        """Train the image transformer from the training set (X)

        Args:
            X (array-like): The training data of shape (n_samples, n_features).
            y: Ignored. Present for continuity with scikit-learn.
            plot (bool): Whether to produce a scatter plot showing the feature
                reduction, hull points, and minimum bounding rectangle.

        Returns:
            self: The fitted instance of the image transformer.
        """
        self._data = X.copy()
        for it in self._its:
            if plot:
                print(it._fe)
            it.fit(X, plot=plot)
            if plot:
                plt.show()
        return self

    def extend_fit(self, feature_extractor):
        """Add additional transformations to an already trained
        MRepImageTransformer instance.

        Args:
            feature_extractor(list): A list of dimensionality reduction class
                instances with a `fit_transform` method returning a 2D array
                of extracted features. Alternatively, a list of tuples where
                the first element is the class instance and the second is
                a discretization option.
        """
        for it_cfg in feature_extractor:
            it = self.initialize_image_transformer(it_cfg)
            it.fit(self._data)
            self._its.append(it)

    def transform(self, X, img_format='rgb', empty_value=0,
                  collate='manifold', return_index=False):
        """Transform the input matrix into image matrices

        Args:
            Args:
            X (array-like): Input matrix of shape (n_samples, n_features) where
                n_features matches the training set.
            img_format (str): The format of the image matrix to return.
                - 'scalar': returns a matrix of shape (n_samples, H, W).
                - 'rgb': returns a ndarray of shape (n_samples, H, W, 3).
                - 'pytorch': returns a tensor of shape (n_samples, 3, H, W).
            empty_value (int or float): Numeric value to fill elements where no
                features are mapped. Default is 0.
            collate (str): The order of the representations.
                - 'manifold': returns all samples sequentially for each feature
                    extractor (manifold).
                - 'sample' returns all representations for each sample grouped
                    together.
                - 'random' returns the representations shuffled using np.random.
            return_index (bool): return two additional arrays. One of the index
                in X for each representation, and one for the index of the
                representation (default=False).

        Returns:
            ndarray or tuple(ndarray): A list of n_samples * n_manifolds numpy
                matrices of dimensions set by the pixel parameter. Optional
                additional ndarrays for the index of the samples and
                representations
        """
        translist = []
        transidx = []
        for idx, it in enumerate(self._its):
            translist.append(it.transform(X, img_format, empty_value))
            transidx.append([idx] * translist[-1].shape[0])

        if collate == 'manifold':
            # keep in order of manifolds
            img_matrices = np.concatenate(translist, axis=0)
            rep_idx = np.concatenate(transidx, axis=0)
            x_index = np.tile(np.arange(X.shape[0]), len(self._its))
        elif collate == 'sample':
            # reorder by sample
            img_shape = translist[0].shape[1:]
            img_matrices = np.stack(translist, axis=1).reshape(-1, *img_shape)
            rep_idx = np.stack(transidx, axis=1).reshape(-1, *img_shape)
            x_index = np.repeat(np.arange(X.shape[0]), len(self._its))
        elif collate == 'random':
            # randomize order
            img_matrices = np.concatenate(translist, axis=0)
            rep_idx = np.concatenate(transidx, axis=0)
            x_index = np.tile(np.arange(X.shape[0]), len(self._its))
            p = np.random.permutation(x_index.shape[0])
            img_matrices = img_matrices[p]
            rep_idx = rep_idx[p]
            x_index = x_index[p]
        else:
            raise ValueError(f"collate method '{collate}' not valid")

        if img_format == 'pytorch':
            img_matrices = torch.from_numpy(img_matrices)
        if return_index:
            return img_matrices, rep_idx, x_index
        else:
            return img_matrices

    def fit_transform(self, X, **kwargs):
        """Train the image transformer from the training set (X) and return
        the transformed data.

        Args:
            X (array-like): Input matrix of shape (n_samples, n_features) where
                n_features matches the training set.
            kwargs: Additional parameters to be passed to the transform method.

        Returns:
            ndarray or tuple(ndarray): A list of n_samples * n_manifolds numpy
                matrices of dimensions set by the pixel parameter. Optional
                additional ndarrays for the index of the samples and
                representations
        """
        self.fit(X)
        img_matrices = self.transform(X, **kwargs)
        return img_matrices

    @staticmethod
    def prediction_reduction(y_hat, index, reduction="mean"):
        """Reduce the prediction score for all representations of a sample
        to a single score.

        Args:
            y_hat (ndarray): The representation prediction score of length
                n_samples * n_manifolds
            index (ndarray): The original sample index for each representation.
            reduction (str): specifies the reduction to apply across
                representations. Options are 'mean' or 'sum'.

        Returns:
            An array of prediction score of length n_samples ordered by index.
        """
        index_set = np.unique(index)
        if reduction == 'mean':
            reduced = np.array([np.mean(y_hat[index == k]) for k in index_set])
        elif reduction == 'sum':
            reduced = np.array([np.sum(y_hat[index == k]) for k in index_set])
        else:
            raise ValueError(f"reduction method '{reduction}' not valid")
        return reduced
