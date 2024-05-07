import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import BisectingKMeans
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import inspect
import warnings

from .utils import asymmetric_greedy_search


class ImageTransformer:
    """Transform features to an image matrix using dimensionality reduction

    This class takes in data normalized between 0 and 1 and converts it to a
    CNN compatible 'image' matrix

    """

    def __init__(self, feature_extractor='tsne', discretization='bin',
                 pixels=(224, 224)):
        """Generate an ImageTransformer instance

        Args:
            feature_extractor: string of value ('tsne', 'pca', 'kpca') or a
                class instance with method `fit_transform` that returns a
                2-dimensional array of extracted features.
            discretization: string of values ('bin', 'assignment'). Defines
                the method for discretizing dimensionally reduced data to pixel
                coordinates.
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
        """
        self._fe = self._parse_feature_extractor(feature_extractor)
        self._dm = self._parse_discretization(discretization)
        self._pixels = self._parse_pixels(pixels)
        self._xrot = np.empty(0)
        self._coords = np.empty(0)

    @staticmethod
    def _parse_pixels(pixels):
        """Check and correct pixel parameter

        Args:
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        return pixels

    @staticmethod
    def _parse_feature_extractor(feature_extractor):
        """Validate the feature extractor value passed to the
        constructor method and return correct method

        Args:
            feature_extractor: string of value ('tsne', 'pca', 'kpca') or a
                class instance with method `fit_transform` that returns a
                2-dimensional array of extracted features.

        Returns:
            function
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
            method: string of value ('bin', 'assignment')

        Returns:
            function
        """
        if method == 'bin':
            func = cls.coordinate_binning
        elif method == 'assignment' or method == 'lsa':
            func = cls.coordinate_optimal_assignment
        elif method == 'ags':
            func = cls.coordinate_heuristic_assignment
        else:
            raise ValueError(f"discretization method '{method}' not valid")
        return func

    @classmethod
    def coordinate_binning(cls, position, px_size):
        """Determine the pixel locations of each feature based on the overlap of
        feature position and pixel locations.

        Args:
            position: a 2d array of feature coordinates
            px_size: tuple with image dimensions

        Returns:
            a 2d array of feature to pixel mappings
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_binned = np.floor(scaled).astype(int)
        # Need to move maximum values into the lower bin
        px_binned[:, 0][px_binned[:, 0] == px_size[0]] = px_size[0] - 1
        px_binned[:, 1][px_binned[:, 1] == px_size[1]] = px_size[1] - 1
        return px_binned

    @staticmethod
    def lsap_optimal_solution(cost_matrix):
        return linear_sum_assignment(cost_matrix)

    @staticmethod
    def lsap_heuristic_solution(cost_matrix):
        return asymmetric_greedy_search(cost_matrix, shuffle=True,
                                        minimize=True)

    @classmethod
    def coordinate_optimal_assignment(cls, position, px_size):
        """Determine the pixel location of each feature using a linear sum
        assignment problem solution on the Euclidean distances between the
        features and the pixels centers'

        Args:
            position: a 2d array of feature coordinates
            px_size: tuple with image dimensions

        Returns:
            a 2d array of feature to pixel mappings
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_centers = cls.calculate_pixel_centroids(px_size)
        # calculate distances
        k = np.prod(px_size)
        clustered = scaled.shape[0] > k
        if clustered:
            dist, labels = cls.clustered_cdist(scaled, px_centers, k)
        else:
            dist = cdist(scaled, px_centers, metric='euclidean')
            labels = np.arange(scaled.shape[0])
        # assignment of features/clusters to pixels
        lsa = cls.lsap_optimal_solution(dist**2)
        px_assigned = np.empty(scaled.shape, dtype=int)
        for i in range(scaled.shape[0]):
            # The feature at i
            # Is mapped to the cluster j=labels[i]
            # Which is mapped to the pixel center px_centers[j]
            # Which is mapped to the pixel k = lsa[1][j]
            # For pixel k, x = k % px_size[0] and y = k // px_size[0]
            j = labels[i]
            ki = lsa[1][j]
            xi = ki % px_size[0]
            yi = ki // px_size[0]
            px_assigned[i] = [yi, xi]
        return px_assigned

    @classmethod
    def coordinate_heuristic_assignment(cls, position, px_size):
        """Determine the pixel location of each feature using a heuristic linear
        assignment problem solution on the Euclidean distances between the
        features and the pixels' centers

        Args:
            position: a 2d array of feature coordinates
            px_size: tuple with image dimensions

        Returns:
            a 2d array of feature to pixel mappings
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_centers = cls.calculate_pixel_centroids(px_size)
        # AGS requires asymmetric assignment so k must be less than pixels
        k = np.prod(px_size) - 1
        clustered = scaled.shape[0] > k
        if clustered:
            dist, labels = cls.clustered_cdist(scaled, px_centers, k)
        else:
            dist = cdist(scaled, px_centers, metric='euclidean')
            labels = np.arange(scaled.shape[0])
        # assignment of features/clusters to pixels
        lsa = cls.lsap_heuristic_solution(dist**2)
        px_assigned = np.empty(scaled.shape, dtype=int)
        for i in range(scaled.shape[0]):
            j = labels[i]
            ki = lsa[1][j]
            xi = ki % px_size[0]
            yi = ki // px_size[0]
            px_assigned[i] = [yi, xi]
        return px_assigned

    @staticmethod
    def calculate_pixel_centroids(px_size):
        """Generate a 2d array of the centroid of each pixel

        Args:
            px_size: tuple with image dimensions

        Returns:
            a 2d array of pixel centroid locations
        """
        px_map = np.empty((np.prod(px_size), 2))
        for i in range(0, px_size[0]):
            for j in range(0, px_size[1]):
                px_map[i * px_size[0] + j] = [i, j]
        px_centroid = px_map + 0.5
        return px_centroid

    @staticmethod
    def clustered_cdist(positions, centroids, k):
        """

        """
        kmeans = BisectingKMeans(n_clusters=k).fit(positions)
        cl_labels = kmeans.labels_
        cl_centers = kmeans.cluster_centers_
        dist = cdist(cl_centers, centroids, metric='euclidean')
        return dist, cl_labels

    def fit(self, X, y=None, plot=False):
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
            plt.scatter(x_new[:, 0], x_new[:, 1], s=1, alpha=0.2)
            plt.fill(x_new[chvertices, 0], x_new[chvertices, 1],
                     edgecolor='r', fill=False)
            plt.fill(mbr[:, 0], mbr[:, 1], edgecolor='g', fill=False)
        return self

    @property
    def pixels(self):
        """The image matrix dimensions

        Returns:
            tuple: the image matrix dimensions (height, width)

        """
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
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

    @staticmethod
    def scale_coordinates(coords, dim_max):
        """Transforms a list of n-dimensional coordinates by scaling them
        between zero and the given dimensional maximum

        Args:
            coords: a 2d ndarray of coordinates
            dim_max: a list of maximum ranges for each dimension of coords

        Returns:
            a 2d ndarray of scaled coordinates
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
        else:
            raise ValueError(f"'{img_format}' not accepted for img_format")
        return img_matrix

    def fit_transform(self, X, **kwargs):
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

    def inverse_transform(self, img):
        """Transform an image layer back to its original space.
            Args:
                img:

            Returns:
                A list of n_samples numpy matrices of dimensions set by
                the pixel parameter
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
        """Generate image matrix with feature counts per pixel

        Returns:
            img_matrix (ndarray): matrix with feature counts per pixel
        """
        fdmat = np.zeros(self._pixels)
        np.add.at(fdmat, tuple(self._coords.T), 1)
        return fdmat

    def coords(self):
        """Get feature coordinates

        Returns:
            ndarray: the pixel coordinates for features
        """
        return self._coords.copy()

    @staticmethod
    def _minimum_bounding_rectangle(hull_points):
        """Find the smallest bounding rectangle for a set of points.

        Modified from JesseBuesking at https://stackoverflow.com/a/33619018
        Returns a set of points representing the corners of the bounding box.

        Args:
            hull_points : nx2 matrix of hull coordinates

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
    def _mat_to_rgb(mat):
        """Convert image matrix to numpy rgb format

        Args:
            mat: {array-like} (..., M, N)

        Returns:
            An numpy.ndarray (..., M, N, 3) with original values repeated
            across RGB channels.
        """

        return np.repeat(mat[..., np.newaxis], 3, axis=-1)
