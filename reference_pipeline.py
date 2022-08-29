from __future__ import print_function

import argparse
import multiprocessing as mp
import os
os.environ["MODIN_ENGINE"] = "ray"
import ray
RUN_ENV = {
    "working_dir": os.getcwd()}  # to make sure every worker can access this pymodule
# # ATTENTION: all functions used in do_worker should be in either pymodule or installed package, do not include
# # functions in current py file, otherwise, it would not be recognized
#

print(f'CPUs: {mp.cpu_count()}')
if ray.is_initialized() == True:
    ray.shutdown()
ray.init(runtime_env=RUN_ENV, num_cpus=mp.cpu_count())
assert ray.is_initialized() == True


import SimpleITK as sitk
import numpy as np
import open3d as o3d
from batchgenerators.utilities.file_and_folder_operations import *

from reference_contour_seg import batch_obtain_mask, batch_obtain_contour
from utils.img_utils import read_nii_bysitk, write_nii_bysitk
from utils.registration_utils import find_cortex_bbox, get_voxel_scaling_matrix

def run_pipeline(ref_path, result_path='/home/binduan/Downloads/NIH/hackathon/results/reference/',
                 down_factor=1):
    # create directory
    maybe_mkdir_p(join(result_path, 'mask'))
    maybe_mkdir_p(join(result_path, 'contour'))
    maybe_mkdir_p(join(result_path, 'point_cloud'))

    if os.path.isfile(ref_path):
        samples = [ref_path]
    else:
        samples = nifti_files(ref_path)
    # assert len(samples) == 1, RuntimeError("Multiple CCF templates detected, please check.")

    name = samples[0].split('/')[-1].split('.nii.gz')[0]

    img_obj, img_info = read_nii_bysitk(samples[0], metadata=True)
    img_np = sitk.GetArrayFromImage(img_obj)  # zyx
    img_info["image_shape"] = img_np.shape  # ZYX

    # downsampling here not used
    # down_factor = None

    print(f'down_factor: {down_factor}')

    axis = int(np.argmax(img_np.shape))

    # run thresholding and save
    print(f'running foreground segmentation')
    masks = batch_obtain_mask(raw_image=img_np, scale_factor=down_factor, axis=axis)
    img_info['mask_volume'] = str(np.sum(masks > 0))  # for initial scaling purpose with subject images

    write_nii_bysitk(join(result_path, 'mask', name + '.nii.gz'), masks, img_obj)

    # run level-set contour
    print(f'finetuning segmentation contour')
    contours = batch_obtain_contour(raw_image=img_np, scale_factor=down_factor, axis=axis)
    contours = np.unique(np.array(contours), axis=0)  # zyx
    contours = np.stack((contours[:, 2], contours[:, 1], contours[:, 0]), axis=-1)  # xyz
    save_pickle(contours, join(result_path, 'contour', name + '.pkl'))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(contours)
    o3d.io.write_point_cloud(join(result_path, 'point_cloud', name + '.pcd'), pcd)

    # pcd = o3d.io.read_point_cloud(join(result_path, 'point_cloud', name + '.pcd'))

    # convert to physical point cloud space
    pcd = pcd.transform(get_voxel_scaling_matrix(img_info['spacing']))

    # cortex mask volume
    _, starting_index, ending_index = find_cortex_bbox(pcd, factor=16, return_indices=True)
    if axis == 0:
        img_info['cortex_mask_volume'] = str(np.sum(masks[starting_index:ending_index + 1, ...] > 0))
    elif axis == 1:
        img_info['cortex_mask_volume'] = str(np.sum(masks[:, starting_index:ending_index + 1, :] > 0))
    else:
        img_info['cortex_mask_volume'] = str(np.sum(masks[..., starting_index:ending_index + 1] > 0))

    # save to file
    save_json(img_info, join(result_path, 'reference.meta'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reference pipeline for 2022 Brain registration hackathon.')
    parser.add_argument('--ref_path', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test1/average_template_10.nii.gz',
                        help='image file (nii.gz).')
    parser.add_argument('--result_path', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test1/reference',
                        help='Path to the image file (nii.gz).')
    parser.add_argument('--down_factor', type=float,
                        default=0.2,
                        help='if specified, subject images is downsampled for faster computation, which might result in inferior performance')

    args = parser.parse_args()

    run_pipeline(ref_path=args.ref_path, result_path=args.result_path, down_factor=args.down_factor)
