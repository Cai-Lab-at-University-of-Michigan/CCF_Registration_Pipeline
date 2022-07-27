from collections import OrderedDict

from skimage.filters.thresholding import _try_all, threshold_li, threshold_isodata, threshold_minimum, \
    threshold_triangle, threshold_otsu, threshold_mean, threshold_yen

import numpy as np
import matplotlib.pyplot as plt


def estimate_radius(img):
    mask = img > threshold_otsu(img)
    coords = np.where(mask)

    h = np.max(coords[0]) - np.min(coords[0])
    w = np.max(coords[1]) - np.min(coords[1])

    radius = np.min([w, h])
    radius = np.min([radius//10, 4])
    return radius


def intersect_ratio(img1, img2):
    agreed = np.logical_and(img1, img2)
    ratio = np.sum(agreed) / np.max([np.sum(img1), np.sum(img2)])
    return ratio


def try_all_threshold(image, figsize=(8, 5), verbose=True):
    """Returns a figure comparing the outputs of different thresholding methods.
    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    figsize : tuple, optional
        Figure size (in inches).
    verbose : bool, optional
        Print function name for each method.
    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    Notes
    -----
    The following algorithms are used:
    * isodata
    * li
    * mean
    * minimum
    * otsu
    * triangle
    * yen
    Examples
    --------
    >>> from skimage.data import text
    >>> fig, ax = try_all_threshold(text(), figsize=(10, 6), verbose=False)
    """

    def thresh(func):
        """
        A wrapper function to return a thresholded image.
        """

        def wrapper(im):
            return im > func(im)

        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
        return wrapper

    # Global algorithms.
    methods = OrderedDict({'Isodata': thresh(threshold_isodata),
                           'Li': thresh(threshold_li),
                           'Mean': thresh(threshold_mean),
                           'Minimum': thresh(threshold_minimum),
                           'Otsu': thresh(threshold_otsu),
                           'Triangle': thresh(threshold_triangle),
                           'Yen': thresh(threshold_yen)})

    return _try_all(image, figsize=figsize,
                    methods=methods, verbose=verbose)
