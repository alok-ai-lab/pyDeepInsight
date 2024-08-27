import numpy as np
import torch
import math
import cv2


def step_blur_kernel(kernel_size, amplification):
    """Generate a square blur kernel with sides of length kernel_size

    Args:
        kernel_size: size of the blur kernel sides
        amplification: the step factor when extended to neighbouring pixel

    Returns:
        A square matrix with sides of kernel_size.
    """
    radius = int((kernel_size-1)/2)
    i, j = np.indices((kernel_size, kernel_size), sparse=True)
    dist = np.sqrt((i-radius)**2 + (j-radius)**2)
    dist[radius, radius] = 1.
    return amplification**(np.ceil(dist)-1)


def apply_blur_kernel(img, kernel):
    """ Applies the kernel to each pixel of an image as a convolution.
    The maximum value for each pixel is returned rather than the sum.
    Values in the original image are not changed. The kernel is applied to
    channels independently and no padding is performed.

    Args:
        img: An PIL formatted (H, W, C) image
        kernel: the kernel to apply.

    Returns:
        An image of the same size as img with the step kernel applied.

    """
    pad0, pad1 = int(kernel.shape[0]//2), int(kernel.shape[1]//2)
    blur_mult = (np.broadcast_to(img.reshape(-1, 1), (img.size, kernel.size))
                 * kernel.flatten()).reshape(*img.shape, *kernel.shape)
    blurred = np.pad(np.zeros_like(img), ((pad0, pad0), (pad1, pad1), (0, 0)))
    valid = np.argwhere(img != 0)
    for a0, a1, a2 in valid:
        a0s = slice(a0, a0+kernel.shape[0])
        a1s = slice(a1, a1+kernel.shape[1])
        blurred[a0s, a1s, a2] = np.maximum(blur_mult[a0, a1, a2],
                                           blurred[a0s, a1s, a2])
    # remove padding
    blurred = blurred[pad0:-pad0, pad1:-pad1]
    # replace original values
    blurred[img > 0] = img[img > 0]
    return blurred


def step_blur(img, kernel_size, amplification=None):
    """ Apply a step blur to an image

    Args:
        img: An PIL formatted (H, W, C) image
        kernel_size: size of the blur kernel sides
        amplification: the step factor when extended to neighbouring pixel

    Returns:
        An image of the same size as img with the step kernel applied.
    """
    if amplification is None:
        amplification = 1/np.sqrt(np.e)
    kernel = step_blur_kernel(kernel_size, amplification)
    return apply_blur_kernel(img, kernel)


class StepBlur2d(torch.nn.Module):

    """A PyTorch nn.module implementation of the step blur images of size
    (N, C, H, W). Based on the blur algorithm of Talla-Chumitaz et al. (2023).
    https://doi.org/10.1016/j.inffus.2022.10.011
    """

    def_amp: float = 1/np.sqrt(np.e)

    def __init__(self, kernel_size, amplification=None):
        """Generate a StepBlur2d instance

        Args:
            kernel_size: size of the blur kernel sides
            amplification: the step factor when extended to neighbouring pixel.
                Default is 1/sqrt(e)
        """
        super().__init__()
        if amplification is None:
            amplification = self.def_amp
        self.kernel = self.step_kernel(kernel_size, amplification)

    def forward(self, img):
        """Apply kernel to image input

        Args:
            img: Torch formatted (N, C, H, W) images

        Returns:
            Images with the step blur kernel applied

        """
        kernel = self.kernel
        padh, padw = int(kernel.shape[-2] // 2), int(kernel.shape[-1] // 2)

        blur_mult = (img.reshape(-1, 1).expand(-1, torch.numel(kernel))
                     * kernel.flatten()).reshape(*img.shape, *kernel.shape)
        blurred = torch.nn.functional.pad(torch.zeros_like(img),
                                          (padw, padw, padh, padh))
        valid = torch.argwhere(img != 0)
        for dn, dc, dh, dw in valid:
            dhs = slice(dh, dh + kernel.shape[-2])
            dws = slice(dw, dw + kernel.shape[-1])
            blurred[dn, dc, dhs, dws] = torch.maximum(blur_mult[dn, dc, dh, dw],
                                                      blurred[dn, dc, dhs, dws])
        # remove padding
        blurred = blurred[:, :, padh:-padh, padw:-padw]
        # replace original values
        blurred[img > 0] = img[img > 0]
        return blurred

    @staticmethod
    def step_kernel(kernel_size, amplification):
        """Generate a square blur kernel with sides of length kernel_size

        Args:
            kernel_size: size of the blur kernel sides
            amplification: the step factor when extended to neighbouring pixel

        Returns:
            A square matrix with sides of kernel_size.
        """
        radius = int((kernel_size - 1) / 2)
        i, j = np.indices((kernel_size, kernel_size), sparse=True)
        dist = np.sqrt((i - radius) ** 2 + (j - radius) ** 2)
        dist[radius, radius] = 1.
        kernel = amplification ** (np.ceil(dist) - 1)
        return torch.tensor(kernel, dtype=torch.float32)


def imgaborfilt(image, wavelength, orientation,
                SpatialFrequencyBandwidth=1, SpatialAspectRatio=0.5):
    """An OpenCV implementation of the MATLAB imgaborfilt as based on the
    StackOverFlow answer https://stackoverflow.com/a/61332913

    Args:
        image: a PIL formatted (H, W, C) image
        wavelength: the wavelength in pixels/cycle of the sinusoidal carrier
        orientation: the orientation of the filter in degrees
        SpatialFrequencyBandwidth: the spatial-frequency bandwidth
        SpatialAspectRatio: Ratio of semimajor and semiminor axes of Gaussian
            envelope

    Return:
        A PIL formatted image after filter application

    """

    orientation = -orientation / 180 * math.pi
    sigma = 0.5 * wavelength * SpatialFrequencyBandwidth
    gamma = SpatialAspectRatio
    shape = 1 + 2 * math.ceil(4 * sigma)  # smaller cutoff is possible for speed
    shape = (shape, shape)
    gabor_filter_real = cv2.getGaborKernel(shape, sigma, orientation,
                                           wavelength, gamma, psi=0)
    gabor_filter_imag = cv2.getGaborKernel(shape, sigma, orientation,
                                           wavelength, gamma, psi=math.pi / 2)
    filtered_image = (cv2.filter2D(image, -1, gabor_filter_real) +
                      1j * cv2.filter2D(image, -1, gabor_filter_imag))
    mag = np.abs(filtered_image)
    phase = np.angle(filtered_image)
    return mag, phase


class GaborFilter2d(torch.nn.Module):
    """A PyTorch nn.module implementation of MATLAB imgaborfilt for a batch
    of images.
    """

    def __init__(self, wavelength=2, orientation=0):
        """Generate a StepBlur2d instance

        Args:
            wavelength: the wavelength in pixels/cycle of the sinusoidal carrier
            orientation: the orientation of the filter in degrees
        """

        super().__init__()
        self.wavelength = wavelength
        self.orientation = orientation

    def forward(self, img):
        """Apply Gabor filter to image input
        Args:
            img: a PyTorch formatted (N, C, H, W) image batch

        Return:
            A PyTorch formatted image after filter application
        """
        pil = self.tensor_to_pil(img)
        gabor_pil = np.zeros_like(pil)
        for j in np.arange(gabor_pil.shape[0]):
            gabor_pil[j] = imgaborfilt(pil[j], self.wavelength,
                                       self.orientation)[0]
        gabor_pil = cv2.normalize(gabor_pil, None, alpha=0., beta=1.,
                                  norm_type=cv2.NORM_MINMAX)
        gabor_img = self.pil_to_tensor(gabor_pil)
        return gabor_img

    @staticmethod
    def pil_to_tensor(img):
        # Convert PIL format to PyTorch format
        return torch.from_numpy(img.transpose(0, 3, 1, 2))

    @staticmethod
    def tensor_to_pil(img):
        # Convert PyTorch format to PIL format
        return img.permute(0, 2, 3, 1).numpy()
