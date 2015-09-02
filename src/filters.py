#!/usr/bin/env python2

""" Contains functions for filtering images.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


def identity_filter(shape):
    """ Creates an identity filter kernel.

        @param shape    Size of the resulting filter (tuple of rows and cols)
        @return         Kernel (numpy array)
    """
    assert(shape[0] % 2 == 0)
    assert(shape[1] % 2 == 0)
    h = np.zeros(shape)
    h[(shape[0] - 1)/2, (shape[1] - 1)/2] = 1
    return h


def gaussian_filter(shape, sigma):
    """ Creates a Gaussian filter kernel.

        @param shape    Size of the resulting filter (tuple of rows and cols)
        @param sigma    Standard deviation of the filter
        @return         Kernel (numpy array)

        Note: the constant 1/(2*pi*sigma**2) in front of the exponential of a
        Gaussian does not matter when constructing the filter because the filter
        must sum to 1 to not change the intensities when convolved with an
        image.

    """
    (m, n) = [(s - 1.)/2. for s in shape]
    (y, x) = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x**2 + y**2)/(2.*sigma**2))
    h_sum = h.sum()
    return h /h_sum if h_sum != 0 else h


def gaussian_filter_seperable(shape, sigma):
    """ Creates row and column vectors of a seperated Gaussian filter.

        @param shape    Size of the resulting filters (tuple of rows and cols)
        @param sigma    Standard deviation of the 
    """
    pass


def plot_filter(h):
    """ Plots the filter in both 2D and 3D.
    """
    h_image = 255.0*(h - np.min(h))/(np.max(h) - np.min(h))
    h_image = h_image.astype('uint8')
    
    # Plot the filter in 2D
    fig = plt.figure()
    fig.canvas.set_window_title('Plot of h')
    ax0 = fig.add_subplot(211) 
    ax0.axis('off')
    h_plot = ax0.imshow(h_image, interpolation='none')
    h_plot.set_cmap('gray')

    # Plot the filter in 3D
    (x, y) = [np.arange(i) for i in h.shape]
    (X, Y) = np.meshgrid(x, y)
    ax1 = fig.add_subplot(212, projection='3d')
    ax1.axis('off')
    surf = ax1.plot_surface(X, Y, h, rstride=1, cstride=1, cmap='gray', linewidth=0, antialiased=False)
    plt.show()


def _test_gaussian_filter():
    shape = (50, 50)
    sigma = 3.0
    h = gaussian_filter(shape, sigma)
    plot_filter(h)
    
    # Read lenna
    lenna_image = Image.open('../datasets/lenna.png')
    lenna = np.array(lenna_image)

    # Convolve the filter with an image
    lenna_blurred = np.zeros_like(lenna)
    lenna_blurred[..., 0] = ndimage.filters.convolve(lenna[:, :, 0], h, mode='constant', cval=0.0)
    lenna_blurred[..., 1] = ndimage.filters.convolve(lenna[:, :, 1], h, mode='constant', cval=0.0)
    lenna_blurred[..., 2] = ndimage.filters.convolve(lenna[:, :, 2], h, mode='constant', cval=0.0)
    lenna_blurred_gt = ndimage.gaussian_filter(lenna, sigma=(sigma, sigma, 0), order=0)

    plt.figure()
    plt.subplot(131)
    plt.imshow(lenna, interpolation='none', cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(lenna_blurred, interpolation='none', cmap='gray')
    plt.title('Custom Gaussian Filter')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(lenna_blurred_gt, interpolation='none', cmap='gray')
    plt.title('Scipy Gaussian Filter')
    plt.axis('off')
    plt.show()


def main():
    _test_gaussian_filter()


if __name__ == '__main__':
    import pdb
    main()
