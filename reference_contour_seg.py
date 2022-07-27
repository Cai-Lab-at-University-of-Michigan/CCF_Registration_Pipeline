import math
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import ray
from skimage import measure
from skimage.filters import median
from skimage.filters.thresholding import threshold_triangle, threshold_mean
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.transform import rescale, resize
from tqdm import tqdm

from lv_set import lv_params, lsf
from utils.batch_utils import batch, step
from utils.coordinate_utils import contours_to_points

WORKERS = mp.cpu_count()

# WORKERS = 64


def median_filtering(image, kernerl_size=10):
    # median filtering
    image = median(image, disk(kernerl_size))

    image = np.interp(image, [np.min(image), np.max(image)], [0, 255]).astype(image.dtype)

    return image


#### thresholding related ####
def obtain_best_threshoding(image: np.ndarray, min_object_size=64, area_threshold=0.1, scale_factor=None):
    """
    :param image:
    :return: foreground of the image
    """

    def thresh_warpper(func, min_object_size=min_object_size, area_threshold=area_threshold, tol=0.2):
        """
        A wrapper function to return a thresholded image.
        """

        def wrapper(im):
            data_type = im.dtype
            im = median_filtering(im)
            im = im > (1 - tol) * func(im)
            im = remove_small_objects(im, min_size=min_object_size, connectivity=im.ndim)
            im = remove_small_holes(im, area_threshold=area_threshold * np.sum(im))
            return np.array(im, dtype=data_type)

        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
        return wrapper

    if np.amax(image) < 15:
        return np.zeros_like(image)

    shape_ = image.shape
    dtype_ = image.dtype

    if scale_factor is not None:
        image = rescale(image, scale_factor, anti_aliasing=False)

    if np.amax(image) > 200:
        mask = thresh_warpper(threshold_triangle)(image)
    else:
        mask = thresh_warpper(threshold_mean)(image)

    if scale_factor is not None:
        mask = resize(mask, shape_, anti_aliasing=False)

    return mask.astype(dtype_)


@ray.remote
def do_thresholding(raw_image, idx, axis=2, scale_factor=None):
    """
    :param image: 2D array
    :return: foreground of the image
    """
    if idx > raw_image.shape[axis]:
        raise RuntimeError('index of the image is wrong')

    if axis == 0:
        image = raw_image[idx, ...]
    elif axis == 1:
        image = raw_image[:, idx, :]
    else:
        image = raw_image[..., idx]

    return obtain_best_threshoding(image, scale_factor=scale_factor)


def batch_obtain_mask(raw_image, axis=2, workers=WORKERS, scale_factor=None):
    """
    :param image: 3D numpy array
    :param fore_axis: which axis to get foreground segmentation
    :param workers: how many workers to process
    :param kwargs:
    :return:
        mask: 3D numpy array
    """
    # global variables
    assert axis in [0, 1, 2], RuntimeError(f'axis should be in [1|2|3], current is {axis}')

    # put shared variable to ray to save memory
    image_ray = ray.put(raw_image)
    axis_ray = ray.put(axis)
    scale_factor_ray = ray.put(scale_factor)

    total = raw_image.shape[axis]

    actual_workers = math.ceil(total / step(total, workers))
    masks = [None] * actual_workers  # initialize empty list

    for w, idx in tqdm(zip(range(workers), batch(list(range(total)), step(total, workers))), total=actual_workers):
        masks[w] = ray.get([do_thresholding.remote(image_ray, i, axis_ray, scale_factor_ray) for i in idx])

    masks = np.stack([m for batch_masks in masks for m in batch_masks], axis=axis)
    return masks


#### contour related #####
def find_contours(image, visualized=False, scale_factor=None):
    shape_ = image.shape

    if scale_factor is not None:
        mask = obtain_best_threshoding(image, scale_factor=scale_factor)
        image = rescale(image, scale_factor, anti_aliasing=False)
        mask = rescale(mask, scale_factor, anti_aliasing=False)
    else:
        mask = obtain_best_threshoding(image)

    # median filtering
    image = median_filtering(image)
    # radius = estimate_radius(image)

    params = lv_params.set_reference_params(image, init_mask=mask, c0=6, kernel_size=3, visualized=visualized)

    phi = lsf.find_lsf(**params)

    if scale_factor is not None:
        phi = resize(phi, shape_, anti_aliasing=True)

    # if visualized:
    #     show_fig.draw_all(phi, params['img'], 10)

    contours = measure.find_contours(phi, 0)

    if visualized:
        image = resize(image, shape_, anti_aliasing=True)
        # show_fig.draw_all(phi, image, 10)
        plt.clf()
        plt.imshow(image, interpolation='nearest', cmap=plt.get_cmap('gray'))
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

        plt.show()
        plt.pause(10)

    return contours


@ray.remote
def do_contour(raw_image, idx, axis, scale_factor=None):
    """
    :param image: 2D array
    :return: foreground of the image
    """
    if idx > raw_image.shape[axis]:
        raise RuntimeError('index of the image is wrong')

    if axis == 0:
        image = raw_image[idx, ...]
    elif axis == 1:
        image = raw_image[:, idx, :]
    else:
        image = raw_image[..., idx]

    return find_contours(image, scale_factor=scale_factor)


def batch_obtain_contour(raw_image, axis=2, workers=WORKERS, scale_factor=None):
    """
    :param image: 3D numpy array
    :param fore_axis: which axis to get foreground segmentation
    :param workers: how many workers to process
    :param kwargs:
    :return:
        points: N x 3 coordinates
    """
    # global variables
    assert axis in [0, 1, 2], RuntimeError(f'axis should be in [0|1|2], current is {axis}')

    # put shared variable to ray to save memory
    image_ray = ray.put(raw_image)
    axis_ray = ray.put(axis)
    scale_factor_ray = ray.put(scale_factor)

    total = raw_image.shape[axis]

    actual_workers = math.ceil(total / step(total, workers))
    contours = [None] * actual_workers  # initialize empty list

    for w, idx in tqdm(zip(range(workers), batch(list(range(total)), step(total, workers))), total=actual_workers):
        contours[w] = ray.get([do_contour.remote(image_ray, i, axis_ray, scale_factor_ray) for i in idx])

    contours = [m for batch_contours in contours for m in batch_contours]

    return contours_to_points(contours, axis=axis)


if __name__ == '__main__':
    from skimage.io import imread

    image = imread('ref.png')
    find_contours(image, visualized=True)
