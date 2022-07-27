import math
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import ray
from skimage import measure
from skimage.filters import median
from skimage.filters.thresholding import threshold_otsu, threshold_triangle, threshold_mean
from skimage.morphology import remove_small_holes, disk, remove_small_objects
from skimage.transform import rescale, resize
from tqdm import tqdm

from lv_set import lv_params, lsf
from utils.batch_utils import batch, step
from utils.coordinate_utils import contours_to_points
from utils.stripe_utils import filter_stripes
from numba import njit, prange

WORKERS = mp.cpu_count()


# WORKERS = 32

# @njit(parallel=True)
def exclude_dim_slices(raw_image):
    for img_dim in range(3):
        to_keep_slices = []
        for slice_no in prange(raw_image.shape[img_dim]):
            if img_dim == 0:
                slice = raw_image[slice_no, ...]
            elif img_dim == 1:
                slice = raw_image[:, slice_no, :]
            else:
                slice = raw_image[..., slice_no]

            if np.amax(slice) > 0.05 * np.amax(raw_image):
                to_keep_slices.append(slice_no)

        assert len(to_keep_slices) > 0

        to_keep_slices = [i for i in range(np.amin(np.array(to_keep_slices)), np.amax(np.array(to_keep_slices)) + 1)]

        if len(to_keep_slices) < raw_image.shape[img_dim] - 20:  # if cut too much, leave some empty slice same as ccf
            max_id, min_id = np.amax(np.array(to_keep_slices)), np.amin(np.array(to_keep_slices))
            to_keep_slices = to_keep_slices + [s + max_id for s in range(10)]
            to_keep_slices = list(reversed([min_id - s for s in range(10)])) + to_keep_slices

            # contain to reasonable idices
            to_keep_slices = list(filter(lambda ss: -1 < ss < raw_image.shape[img_dim], to_keep_slices))

        to_keep_slices = np.array(to_keep_slices)

        if img_dim == 0:
            raw_image = raw_image[to_keep_slices, ...]
        elif img_dim == 1:
            raw_image = raw_image[:, to_keep_slices, :]
        else:
            raw_image = raw_image[..., to_keep_slices]

    return raw_image


def remove_bright_lines(image, kernerl_size=20):
    # pad image for removing bright lines
    pad_image = np.pad(image, pad_width=5)
    # median filtering
    pad_image = median(pad_image, disk(kernerl_size))
    image = pad_image[5:-5, 5:-5]

    image = np.interp(image, [np.min(image), np.max(image)], [0, 255]).astype(image.dtype)

    return image


#### thresholding related #####
def obtain_best_threshoding(image: np.ndarray, stripe=False, fft=False, kernel_size=20, min_object_size=128,
                            area_threshold=0.2,
                            scale_factor=None, tol=0.2):
    """
    :param image: 2D array
    :return: foreground of the image
    """

    # if np.amax(image) < 15:
    #     return np.zeros_like(image)

    # pe = np.percentile(image.flatten(), q=99)
    # if pe < 60:
    #     image[image > pe] = pe

    shape_ = image.shape
    dtype_ = image.dtype

    if scale_factor is not None:
        image = rescale(image, scale_factor, anti_aliasing=False)
        kernel_size = kernel_size * np.amin(scale_factor)

    if stripe:
        image = remove_bright_lines(image, kernerl_size=kernel_size)

    if np.amax(image) < 10:
        return np.zeros_like(image)

    # remove striples and thresholding
    if fft:
        fimg = filter_stripes(image)
        if threshold_otsu(fimg) > 5 * threshold_mean(fimg) or threshold_otsu(fimg) > 5 * threshold_triangle(fimg):
            if threshold_otsu(fimg) < 50:
                thresh = threshold_otsu(fimg)
            elif threshold_triangle(fimg) > 2 * threshold_mean(fimg):
                fimg = remove_bright_lines(fimg, kernerl_size=kernel_size)
                thresh = threshold_mean(fimg)
            else:
                thresh = max(threshold_triangle(fimg), threshold_mean(fimg))
        else:
            thresh = max(threshold_triangle(fimg), threshold_otsu(fimg))

        mask = fimg > (1 - tol) * thresh
    else:
        if threshold_otsu(image) > 5 * threshold_mean(image) or threshold_otsu(image) > 5 * threshold_triangle(image):
            if threshold_otsu(image) < 50:
                thresh = threshold_otsu(image)
            elif threshold_triangle(image) > 2 * threshold_mean(image):
                image = remove_bright_lines(image, kernerl_size=kernel_size)
                thresh = threshold_mean(image)
            else:
                thresh = max(threshold_triangle(image), threshold_mean(image))
        else:
            thresh = max(threshold_triangle(image), threshold_otsu(image))
        mask = image > (1 - tol) * thresh

    # remove small holes
    mask = remove_small_objects(mask, min_size=min_object_size, connectivity=mask.ndim)

    area_threshold = area_threshold * np.sum(mask)
    mask = remove_small_holes(mask, area_threshold=area_threshold)

    if scale_factor is not None:
        mask = resize(mask, shape_, anti_aliasing=False)

    return mask.astype(np.uint8)


@ray.remote
def do_thresholding(raw_image, idx, axis=0, scale_factor=None):
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


def batch_obtain_mask(raw_image, axis=0, workers=WORKERS, scale_factor=None):
    """
    :param image: 3D numpy array
    :param fore_axis: which axis to get foreground segmentation
    :param workers: how many workers to process
    :param kwargs:
    :return:
        mask: 3D numpy array
    """
    # global variables
    assert axis in [0, 1, 2], RuntimeError(f'axis should be in [0|1|2], current is {axis}')

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
def find_contours(image, visualized=False, scale_factor=None, stripe=False):
    """return contours list"""

    shape_ = image.shape

    if scale_factor is not None:
        mask = obtain_best_threshoding(image, scale_factor=scale_factor)
        image = rescale(image, scale_factor, anti_aliasing=False)
        mask = rescale(mask, scale_factor, anti_aliasing=False)

        image = np.array(image * 255, dtype=np.uint8)
    else:
        mask = obtain_best_threshoding(image)

    if stripe:
        image = remove_bright_lines(image)

    # radius = estimate_radius(image)
    params = lv_params.set_subject_params(image, init_mask=mask, c0=6, kernel_size=3, visualized=visualized)

    phi = lsf.find_lsf(**params)

    if scale_factor is not None:
        phi = resize(phi, shape_, anti_aliasing=True)

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
def do_contour(raw_image, idx, axis=0, scale_factor=None):
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

    if np.amax(image) <= 1:
        return []  # return empty list due to all background

    return find_contours(image, scale_factor=scale_factor)


def batch_obtain_contour(raw_image, axis=0, workers=WORKERS, scale_factor=None):
    """
    :param image: 3D numpy array
    :param fore_axis: which axis to get foreground segmentation
    :param workers: how many workers to process
    :param kwargs:
    :return:
        points: N x 3 coordinates
    """
    # global variables
    assert axis in [0, 1, 2], RuntimeError(f'axis should be in [1|2|3], current is {axis}')

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
    import numpy as np

    target_spacing = np.array((0.02500000037252903, 0.02500000037252903, 0.02500000037252903))
    src_spacing = np.array((0.009600000455975533, 0.009600000455975533, 0.00800000037997961))

    scale_factor = src_spacing / target_spacing
    scale_factor = scale_factor[1:]

    image = imread('sub.png')
    find_contours(image, visualized=True, scale_factor=scale_factor)
