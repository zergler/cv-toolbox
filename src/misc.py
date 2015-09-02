#!/usr/bin/python2


""" Contains useful miscellaneous functions.
"""

import numpy as np


def rgb2gray(image, weights):
    """ Converts an rgb image to grayscale.
    """
    return np.dot(image, [0.299, 0.587, 0.144])
