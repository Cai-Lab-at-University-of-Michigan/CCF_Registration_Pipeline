from __future__ import print_function

import argparse
import glob
import multiprocessing as mp
import os
os.environ["MODIN_ENGINE"] = "ray"
import ray

RUN_ENV = {
    "working_dir": os.getcwd()}  # to make sure every worker can access this pymodule

# ATTENTION: all functions used in do_worker should be in either pymodule or installed package, do not include
# functions in current py file, otherwise, it would not be recognized

ray.shutdown()
ray.init(runtime_env=RUN_ENV, num_cpus=mp.cpu_count())
assert ray.is_initialized() == True

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from batchgenerators.utilities.file_and_folder_operations import *

from subject_contour_seg import batch_obtain_mask, batch_obtain_contour
from utils.img_utils import read_nii_bysitk, write_nii_bysitk
from utils.registration_utils import find_cortex_bbox, get_voxel_scaling_matrix, \
    iterative_estimate_transformation_by_cortex, calculate_warped_shape, \
    prepare_affine_matrix, warp_affine3d
from memory_profiler import profile


def run_pipeline(sub_path='/home/binduan/Downloads/NIH/hackathon/results/subject/preprocessed/',
                 result_path='/home/binduan/Downloads/NIH/hackathon/results/subject/',
                 ref_path='/home/binduan/Downloads/NIH/hackathon/results/reference/',
                 ref_meta_file='/home/binduan/Downloads/NIH/hackathon/results/reference/reference.meta',
                 down_factor=None):
    # create directory
    maybe_mkdir_p(join(result_path, 'meta'))
    maybe_mkdir_p(join(result_path, 'mask'))
    maybe_mkdir_p(join(result_path, 'contour'))
    maybe_mkdir_p(join(result_path, 'point_cloud'))

    maybe_mkdir_p(join(result_path, 'interim'))  # path save interim results
    maybe_mkdir_p(join(result_path, 'prealigned'))  # path save pre-aligned image

    if os.path.isfile(sub_path):
        samples = [sub_path]
    else:
        samples = nifti_files(sub_path)
    # assert len(samples) == 132, RuntimeError("Not enought subjects detected, please check.")

    # read reference meta_data
    ref_info = load_json(ref_meta_file)
    ref_spacing = np.array(ref_info['spacing'])
    ref_size = np.array(ref_info['image_shape'])

    # load ref point cloud
    ref_pcd = o3d.io.read_point_cloud(glob.glob(join(ref_path, 'point_cloud', '*.pcd'))[0])
    ref_pcd = ref_pcd.transform(get_voxel_scaling_matrix(ref_spacing))

    # ref_masks = sitk.GetArrayFromImage(read_nii_bysitk(glob.glob(join(ref_path, 'mask', '*.nii.gz'))[0]))
    ref_cortex = find_cortex_bbox(ref_pcd, factor=18.)

    for i, sample in enumerate(samples):

        name = sample.split('/')[-1].split('.nii.gz')[0]

        # if os.path.isfile(join(result_path, 'point_cloud', name + '.pcd')):
        #     continue

        img_obj, img_info = read_nii_bysitk(sample, metadata=True)
        img_np = sitk.GetArrayFromImage(img_obj)

        # downsampling here not using for now
        # down_factor = None

        # only one step is needed, bounding boxes of cortex is aligned iteratively though
        for step in range(1):
            if step < 1:
                axis = int(np.argmax(img_np.shape))
            else:
                axis = np.argmax(np.array(ref_size))

            if step > 0:
                last_cortex_width = ending_index - starting_index

            write_nii_bysitk(join(result_path, 'interim', name + f'_isotropical_step_{step}_raw.nii.gz'), img_np, img_obj)

            # run thresholding and save
            print(f'{i:03d}: step-{step} running foreground segmentation')
            masks = batch_obtain_mask(raw_image=img_np, scale_factor=down_factor, axis=axis)

            write_nii_bysitk(join(result_path, 'interim', name + f'_isotropical_step_{step}_mask.nii.gz'),
                             np.array(masks, dtype=np.uint8), img_obj)

            del masks

            # run level-set contour
            print(f'{i:03d}: step-{step} finetuning segmentation contour')
            contours = batch_obtain_contour(raw_image=img_np, scale_factor=down_factor, axis=axis)
            contours = np.unique(np.array(contours), axis=0)  # zyx
            contours = np.stack((contours[:, 2], contours[:, 1], contours[:, 0]), axis=-1)  # xyz
            save_pickle(contours, join(result_path, 'interim', name + f'_isotropical_step_{step}_contour.pkl'))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(contours)
            o3d.io.write_point_cloud(join(result_path, 'interim', name + f'_isotropical_step_{step}_pcd.pcd'), pcd)

            pcd_shape = pcd.get_max_bound() - pcd.get_min_bound()
            print(pcd_shape)

            pcd = pcd.transform(get_voxel_scaling_matrix(ref_spacing))
            try:
                sub_cortex, starting_index, ending_index = find_cortex_bbox(pcd, factor=16., return_indices=True)
            except:
                print("finding cortex wrong due to bad segmentation, please inspect.")
                continue

            if step > 0:
                if ending_index - starting_index > 1.05 * last_cortex_width or ending_index - starting_index < 0.95 * last_cortex_width:
                    print("Too many difference between two steps, using the last step result")
                    break

            # apply affine to img_np and repeat
            print(f'{i:03d}: step-{step} applying affine to img_np and repeating')
            scaling_affine = get_voxel_scaling_matrix(np.array(ref_spacing))

            mid_affine = iterative_estimate_transformation_by_cortex(ref_cortex, sub_cortex)
            # draw_registration_result(sub_cortex, ref_cortex)
            # draw_registration_result(pcd, ref_pcd, compose_affine_transforms(mid_affine))

            aff = prepare_affine_matrix(scaling_affine, mid_affine, scaling_affine)

            np.savetxt(join(result_path, 'affines', name + '_cortex.txt'), aff)

            dst_shape = calculate_warped_shape(img_np.shape, aff)

            img_np = np.array(img_np / 255., dtype=np.float32)  # zyx

            # To Tensor
            img_np = torch.from_numpy(img_np)
            img_np = img_np[None, None, ...]  # BCZYX

            aff = torch.from_numpy(aff[:3, :])
            aff = aff[None, ...]

            print(f'target shape:{dst_shape}')

            warped_image = warp_affine3d(img_np, aff, dsize=dst_shape)

            warped_image = warped_image.numpy()
            warped_image = np.array(warped_image * 255, dtype=np.uint8)
            warped_image = np.squeeze(warped_image)

            warped_image = np.interp(warped_image, [np.min(warped_image), np.max(warped_image)], [0, 255]).astype(
                np.uint8)

            img_obj = sitk.GetImageFromArray(warped_image)
            img_obj.SetSpacing(ref_spacing)

            img_np = warped_image

            print(f'One iteration of alignment finished.')

        # final step
        write_nii_bysitk(join(result_path, 'prealigned', name + '.nii.gz'), img_np, img_obj)

        axis = int(np.argmax(img_np.shape))
        # mask
        print(f'{i:03d}: step-final running foreground segmentation')
        masks = batch_obtain_mask(raw_image=img_np, scale_factor=down_factor, axis=axis)

        write_nii_bysitk(join(result_path, 'mask', name + '.nii.gz'), np.array(masks, dtype=np.uint8), img_obj)

        print(f'{i:03d}: step-final finetuning segmentation contour')
        contours = batch_obtain_contour(raw_image=img_np, scale_factor=down_factor, axis=axis)
        contours = np.unique(np.array(contours), axis=0)  # zyx
        contours = np.stack((contours[:, 2], contours[:, 1], contours[:, 0]), axis=-1)  # xyz
        save_pickle(contours, join(result_path, 'contour', name + '.pkl'))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(contours)
        o3d.io.write_point_cloud(join(result_path, 'point_cloud', name + '.pcd'), pcd)

        img_info['image_shape'] = img_np.shape
        img_info['spacing'] = img_obj.GetSpacing()
        img_info['mask_volume'] = str(np.sum(masks > 0))
        try:
            _, starting_index, ending_index = find_cortex_bbox(pcd, factor=16, return_indices=True)
            if axis == 0:
                img_info['cortex_mask_volume'] = str(np.sum(masks[starting_index:ending_index + 1, ...] > 0))
            elif axis == 1:
                img_info['cortex_mask_volume'] = str(np.sum(masks[:, starting_index:ending_index + 1, :] > 0))
            else:
                img_info['cortex_mask_volume'] = str(np.sum(masks[..., starting_index:ending_index + 1] > 0))
        except:
            print("final step finding cortex goes wrong, skip")
        finally:
            save_json(img_info, join(result_path, 'meta', name + '.meta'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reference pipeline for 2022 Brain registration hackathon.')
    parser.add_argument('--sub_path', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test1/subject/preprocessed/',
                        help='Path to the preprocessed image file (nii.gz).')
    parser.add_argument('--result_path', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test1/subject/',
                        help='Path for storing result.')
    parser.add_argument('--ref_path', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test1/reference/',
                        help='Path to the reference folder.')
    parser.add_argument('--ref_meta_file', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test1/reference/reference.meta',
                        help='Path to the reference meta file.')
    parser.add_argument('--down_factor', type=float,
                        default=0.2,
                        help='if specified, subject images is downsampled for faster computation, which might result in inferior performance')

    args = parser.parse_args()

    run_pipeline(sub_path=args.sub_path, result_path=args.result_path, ref_path=args.ref_path,
                 ref_meta_file=args.ref_meta_file,
                 down_factor=args.down_factor)
