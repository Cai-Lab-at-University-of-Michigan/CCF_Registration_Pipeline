"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""

import numpy as np
import skimage.io

from .drlse_algo import SINGLE_WELL, DOUBLE_WELL
from skimage.morphology import disk, binary_dilation
import matplotlib.pyplot as plt


def set_reference_params(img, init_mask, c0=2, kernel_size=10, visualized=False):
    # dilation mask
    init_mask = binary_dilation(init_mask, disk(kernel_size))

    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = -c0
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[init_mask] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 100,  # time step
        'iter_inner': 5,
        'iter_outer': 10,
        'lmda': 1,  # coefficient of the weighted length term L(phi)
        'alfa': -4,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
        'visualized': visualized,
    }


def set_subject_params(img, init_mask, c0=2, kernel_size=15, visualized=False):
    # dilation mask
    init_mask = binary_dilation(init_mask, disk(kernel_size))

    # initialize LSF as binary step function
    c0 = -c0
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[init_mask] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 20,  # time step
        'iter_inner': 5,
        'iter_outer': 5,
        'lmda': 2,  # coefficient of the weighted length term L(phi)
        'alfa': -4,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
        'visualized': visualized,
    }