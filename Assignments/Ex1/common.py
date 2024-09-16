import numpy as np

from scipy.ndimage import gaussian_filter

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    return np.mean(I, axis=2) # Placeholder

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    kernel = np.array([0.5, 0, -0.5])

    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)

    # Convolve the kernel with each row for horizontal gradient
    for i in range(I.shape[0]):
        Ix[i, :] = np.convolve(I[i, :], kernel, mode='same')

    # Convolve the kernel with each column for vertical gradient
    for i in range(I.shape[1]):
        Iy[:, i] = np.convolve(I[:, i], kernel, mode='same')

    # Compute the gradient magnitude
    Im = np.sqrt(Ix**2 + Iy**2)

    return Ix, Iy, Im

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.

    result = gaussian_filter(I, sigma=sigma)
    return result

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    # Get the indices of the pixels that are above the threshold
    y_coords, x_coords = np.nonzero(Im > threshold)
    
    # Compute the gradient orientation for each edge pixel
    angles = np.arctan2(Iy[y_coords, x_coords], Ix[y_coords, x_coords])

    return x_coords, y_coords, angles
