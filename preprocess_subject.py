from __future__ import print_function

import argparse

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from numba import njit, prange
from skimage.transform import rescale, resize
from tqdm import tqdm
from subject_contour_seg import exclude_dim_slices

from utils.img_utils import read_nii_bysitk, write_nii_bysitk


@njit
def padding(img, pad):
    padded_img = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, img.shape[2] + 2 * pad))
    padded_img[pad:-pad, pad:-pad, pad:-pad] = img
    return padded_img


@njit(parallel=True)
def AdaptiveMedianFilter(img, s=3, sMax=10):
    D, H, W = img.shape
    a = sMax // 2
    padded_img = padding(img, a)

    f_img = np.zeros(padded_img.shape)

    for i in prange(D + sMax - s + 1):
        for j in range(H + sMax - s + 1):
            for k in range(W + sMax - s + 1):
                value = Lvl_A(padded_img, i, j, k, s, sMax)
                f_img[i, j, k] = value

    return f_img[a:-a, a:-a, a:-a]


@njit
def Lvl_A(mat, x, y, z, s, sMax):
    window = mat[x:x + s + 1, y:y + s + 1, z:z + s + 1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return Lvl_B(window, Zmin, Zmed, Zmax)
    else:
        s += 2
        if s <= sMax:
            return Lvl_A(mat, x, y, z, s, sMax)
        else:
            return Zmed


@njit
def Lvl_B(window, Zmin, Zmed, Zmax):
    d, h, w = window.shape

    Zxyz = window[d // 2, h // 2, w // 2]
    B1 = Zxyz - Zmin
    B2 = Zxyz - Zmax

    if B1 > 0 and B2 < 0:
        return Zxyz
    else:
        return Zmed


# @ray.remote
def downsample_and_remove_bright_lines(sample, result_path, ref_meta_file, kernerl_size=3):
    ref_info = load_json(ref_meta_file)
    ref_spacing = np.array(ref_info['spacing'])
    maybe_mkdir_p(join(result_path, 'affines'))

    name = sample.split('/')[-1].split('.nii.gz')[0]

    if os.path.isfile(join(result_path, 'preprocessed', name + '.nii.gz')):
        return

    print("start downsampling")
    img_obj, img_info = read_nii_bysitk(sample, metadata=True)
    img_np = sitk.GetArrayFromImage(img_obj)

    # downsample the subject image into scale of reference image first
    sub_spacing = np.array(img_info['spacing'])

    # make it isotropical based on physical spacing
    scales = np.array([1.] * 3)
    if np.amax(sub_spacing) > np.amin(sub_spacing):
        scales = np.where(sub_spacing == np.amin(sub_spacing), np.amin(sub_spacing) / np.amax(sub_spacing), 1)

        sub_spacing = np.array([np.amax(sub_spacing)] * 3)

    scales = (sub_spacing / ref_spacing) * scales
    scales = np.array([scales[2], scales[1], scales[0]])  # zyx

    aff = np.eye(4)
    aff[0, 0] = scales[0]
    aff[1, 1] = scales[1]
    aff[2, 2] = scales[2]

    np.savetxt(join(result_path, 'affines', name + '_pre_scaling.txt'), aff)

    img_np = rescale(img_np, scales, anti_aliasing=False)

    img_np = np.interp(img_np, [np.min(img_np), np.max(img_np)], [0, 255]).astype(np.uint8)

    # exclude dark slices among each dimension
    img_np = exclude_dim_slices(img_np)

    img_obj = sitk.GetImageFromArray(img_np)
    img_obj.SetSpacing(ref_spacing)

    write_nii_bysitk(join(result_path, 'downsampled', name + '.nii.gz'), img_np, img_obj)

    # median filtering
    print('start median filtering')
    fimage = rescale(np.array(img_np / 255., dtype=np.float32), 0.5, anti_aliasing=False)
    fimage = AdaptiveMedianFilter(fimage, kernerl_size)
    fimage = resize(fimage, img_np.shape, anti_aliasing=True)

    img_np = np.interp(fimage, [np.min(fimage), np.max(fimage)], [0, 255]).astype(np.uint8)

    img_obj = sitk.GetImageFromArray(img_np)
    img_obj.SetSpacing(ref_spacing)

    write_nii_bysitk(join(result_path, 'preprocessed', name + '.nii.gz'), img_np, img_obj)

    return


def run_pipeline(sub_path, result_path, ref_meta_file):
    # create directory
    maybe_mkdir_p(join(result_path, 'downsampled'))
    maybe_mkdir_p(join(result_path, 'preprocessed'))

    downsample_and_remove_bright_lines(sub_path, result_path, ref_meta_file)

    # ray.get([remove_bright_lines.remote(sample, result_path) for sample in samples])
    print("Preprocessing Step Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reference pipeline for 2022 Brain registration hackathon.')
    parser.add_argument('--sub_path', type=str,
                        default='/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/subject/',
                        help='Path to the raw image file (nii.gz).')
    parser.add_argument('--result_path', type=str,
                        default='/home/binduan/Downloads/NIH/hackathon/results/subject/',
                        help='Path to the image file (nii.gz).')
    parser.add_argument('--ref_meta_file', type=str,
                        default='/home/binduan/Downloads/NIH/hackathon/results/reference/reference.meta',
                        help='Path to the reference meta file.')

    args = parser.parse_args()

    run_pipeline(sub_path=args.sub_path, result_path=args.result_path, ref_meta_file=args.ref_meta_file)
