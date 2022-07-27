import argparse
import copy
import os.path
import time

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from batchgenerators.utilities.file_and_folder_operations import *

from utils.img_utils import write_nii_bysitk, read_nii_bysitk
from utils.registration_utils import GlobalContourRegister as GCR
from utils.registration_utils import compose_affine_transforms, prepare_affine_matrix
from utils.registration_utils import execute_global_registration, execute_fast_global_registration, \
    iterative_estimate_transformation, totuple, draw_registration_result, warp_affine3d, calculate_warped_shape

from utils.ants_utils import performAntsRegistration
import torch.nn.functional as F
from skimage.transform import rescale
import scipy.io as sio
import scipy.linalg


# from kornia.geometry import warp_affine3d

def step_affine(ref_sample, ref_meta_file, sub_sample, sub_meta, downsampled_voxel_size, max_corr_distance,
                result_path):
    ref_GCR = GCR(ref_sample, ref_meta_file, downsampled_voxel_size, isref=True)
    ref_pcd = ref_GCR.load_point_cloud()
    ref_pcd_down, ref_pcd_fpfh = ref_GCR.compute_fpfh_features(ref_pcd)

    sub_GCR = GCR(sub_sample, sub_meta, downsampled_voxel_size, ref_meta=ref_meta_file)
    sub_pcd = sub_GCR.load_point_cloud()

    # sub_pcd_copy = copy.deepcopy(sub_pcd)

    step_aff = np.eye(4)
    mid_affines = []

    # iterative apply
    for step in range(4):
        sub_pcd.transform(step_aff)

        sub_pcd_down, sub_pcd_fpfh = sub_GCR.compute_fpfh_features(sub_pcd)

        print(f'ref_pcd: {ref_pcd}, ref_pcd_down: {ref_pcd_down}')
        print(f'sub_pcd: {sub_pcd}, sub_pcd_down: {sub_pcd_down}')

        ############### Step 0: Init Global registration ######################
        # mid_affine = iterative_estimate_transformation(ref_pcd, sub_pcd)
        # sub_pcd_down.transform(compose_affine_transforms(mid_affine))
        # draw_registration_result(sub_pcd_down, ref_pcd_down)
        # mid_affine = [np.eye(4)]  # we do this earlier in subject pipeline, so identity transform here
        # draw_registration_result(sub_pcd_down, ref_pcd_down)

        ############### Step 1: RANSAC global registration ####################
        print("Roughly global registration")
        start = time.time()
        result_ransac = execute_global_registration(sub_pcd_down, ref_pcd_down,
                                                    sub_pcd_fpfh, ref_pcd_fpfh,
                                                    voxel_size=downsampled_voxel_size)
        print("Global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        # draw_registration_result(sub_pcd_down, ref_pcd_down, result_ransac.transformation)

        ################ Step 2: Point-to-point ICP Refinement ##################
        print("Apply point-to-point ICP")
        start = time.time()
        reg_p2p = o3d.pipelines.registration.registration_icp(
            sub_pcd_down, ref_pcd_down, max_corr_distance, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.9, relative_rmse=1e-5,
                                                              max_iteration=1000000))
        print("Point-to-point ICP took %.3f sec.\n" % (time.time() - start))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(sub_pcd_down, ref_pcd_down, reg_p2p.transformation)

        step_aff = reg_p2p.transformation

        # save global affine
        mid_affines.append(reg_p2p.transformation)

    aff = prepare_affine_matrix(sub_GCR.scaling_matrix, mid_affines, ref_GCR.scaling_matrix)

    # draw_registration_result(sub_pcd_copy, ref_pcd, compose_affine_transforms(mid_affines))

    name = sub_sample.split('/')[-1].split('.pcd')[0]
    np.savetxt(join(result_path, 'affines', name + '_pcd.txt'), aff)

    print(f'final affine: {aff}')

    return aff


def step_warp_affine(aff, sub_img, result_path):
    ################ Step 3: Warping image with affine transform ################
    name = sub_img.split('/')[-1].split('.nii.gz')[0]

    img_obj = read_nii_bysitk(sub_img, metadata=False)
    image = sitk.GetArrayFromImage(img_obj)
    image = image / 255.  # zyx
    # image = rescale(image, scale=0.125)

    dst_shape = calculate_warped_shape(image.shape, aff)

    # To Tensor
    image = torch.from_numpy(image)
    image = image[None, None, ...]  # BCZYX
    # aff[np.diag_indices_from(aff)] = 8
    aff = torch.from_numpy(aff[:3, :])
    aff = aff[None, ...]

    warped_image = warp_affine3d(image, aff, dsize=dst_shape)

    warped_image = warped_image.numpy()
    warped_image = np.array(warped_image * 255, dtype=np.uint8)

    warped_image_path = join(result_path, name + '.nii.gz')
    write_nii_bysitk(warped_image_path, warped_image)

    return warped_image_path


def step_non_parametric_registration(aff, sub_img, ref_img, result_path):
    ################ Step 4: Non-parametric Image Registration ################
    pass


def run_pipeline(ref_img_path, ref_pcd_path, ref_meta_file, sub_img_path, sub_pcd_path, sub_meta_path, result_path,
                 max_corr_distance,
                 downsampled_voxel_size, factor):
    """
    || step 0: estimate init global registration to align rotation, size, center
    || Step 1: RANSAC and point to point for global registration
    || Step 2: Local registration to align firmly
    """
    # create directory
    maybe_mkdir_p(join(result_path, 'affines'))
    maybe_mkdir_p(join(result_path, 'warped_image_affine'))
    maybe_mkdir_p(join(result_path, 'warped_image_ants_affine'))
    maybe_mkdir_p(join(result_path, 'deformed'))

    if os.path.isfile(ref_img_path):
        ref_imgs = [ref_img_path]
    else:
        ref_imgs = subfiles(ref_img_path, suffix='.nii.gz')
    
    if os.path.isfile(ref_pcd_path):
        ref_samples = [ref_pcd_path]
    else:
        ref_samples = subfiles(ref_pcd_path, suffix='pcd')

    assert len(ref_samples) == 1, RuntimeError("Multiple CCF templates detected, please check.")

    if os.path.isfile(sub_img_path):
        sub_imgs = [sub_img_path]
    else:
        sub_imgs = subfiles(sub_img_path, suffix='.nii.gz')

    if os.path.isfile(sub_pcd_path):
        sub_samples = [sub_pcd_path]
    else:
        sub_samples = subfiles(sub_pcd_path, suffix='pcd')
    # assert len(sub_samples) == 132, RuntimeError("Not enough subjects detected, please check.")

    if os.path.isfile(sub_meta_path):
        sub_metas = [sub_meta_path]
    else:
        sub_metas = subfiles(sub_meta_path, suffix='meta')
    # assert len(sub_metas) == 132, RuntimeError("Not enough subjects detected, please check.")

    # load reference image contours and info
    ref_img_shape = totuple(load_json(ref_meta_file)['image_shape'])
    ref_sample = ref_samples[0]

    for (sub_img, sub_sample, sub_meta) in zip(sub_imgs, sub_samples, sub_metas):

        name = sub_img.split('/')[-1].split('.nii.gz')[0]

        if not os.path.isfile(join(result_path, 'warped_image_affine', name + '.nii.gz')):
            print(f'Start registerinng sample: {sub_img.split("/")[-1]}')
            aff = step_affine(ref_sample, ref_meta_file, sub_sample, sub_meta, downsampled_voxel_size,
                              max_corr_distance,
                              result_path)
            step_warp_affine(aff, sub_img, join(result_path, 'warped_image_affine'))

        # # using ants
        # if not os.path.isfile(join(result_path, 'deformed', name + '.nii.gz')):
        #     print(f'using ants to register sample: {sub_img.split("/")[-1]}')
        #     record_path = join(result_path, 'deformed')
        #     affine_warped_image = join(result_path, 'warped_image_affine', name + '.nii.gz')
        #     warpedfixout, warpedmovout, loutput = performAntsRegistration(mv_path=affine_warped_image,
        #                                                                   target_path=ref_imgs[0],
        #                                                                   init_aff=init_aff,
        #                                                                   tl_path=ref_ann_path,
        #                                                                   registration_type='syn',
        #                                                                   record_path=record_path, fname=name,
        #                                                                   outprefix=join(record_path, 'temp'))
        #
        #     write_nii_bysitk(join(record_path, name + '_warped_ccf.nii.gz'), warpedfixout)
        #     write_nii_bysitk(join(record_path, name + '_warped.nii.gz'), warpedmovout)
        #     write_nii_bysitk(join(record_path, name + '_warped_ccf_annotation.nii.gz'), loutput)

            # aff = sio.loadmat(join(record_path, name + '_affine.mat'))
            # ants_aff = np.reshape(aff['AffineTransform_float_3_3'], (4, 3))
            # trans = np.eye(4)
            # trans[:3, :3] = ants_aff[:3, :3]
            # trans[:3, -1:] = np.transpose(ants_aff[-1:, :])
            #
            # aff = scipy.linalg.inv(trans)
            #
            # np.savetxt(join(result_path, 'affines', name + '_ants.txt'), aff)
            #
            # step_warp_affine(aff, sub_img, join(result_path, 'warped_image_ants_affine'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reference pipeline for 2022 Brain registration hackathon.')
    parser.add_argument('--ref_img_path', type=str,
                        default='/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/reference/',
                        help='Path to the image file (nii.gz).')
    parser.add_argument('--ref_ann_path', type=str,
                        default='/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/reference/annotation_25.nii.gz',
                        help='Path to the ccf annotation file (nii.gz).')
    parser.add_argument('--ref_pcd_path', type=str,
                        default='/home/binduan/Downloads/NIH//hackathon/results/reference/point_cloud',
                        help='Path to the point cloud file (pcd).')
    parser.add_argument('--ref_meta_file', type=str,
                        default='/home/binduan/Downloads/NIH/hackathon/results/reference/reference.meta',
                        help='Path to the reference meta file.')
    parser.add_argument('--sub_img_path', type=str,
                        default='/home/binduan/Downloads/NIH//hackathon/results/subject/prealigned/',
                        help='Path to the raw image file (.nii.gz).')
    parser.add_argument('--sub_pcd_path', type=str,
                        default='/home/binduan/Downloads/NIH//hackathon/results/subject/point_cloud/',
                        help='Path to the pcd file (pcd).')
    parser.add_argument('--sub_meta_path', type=str,
                        default='/home/binduan/Downloads/NIH/hackathon/results/subject/meta/',
                        help='Path to the subject meta files.')
    parser.add_argument('--result_path', type=str,
                        default='/home/binduan/Downloads/NIH/hackathon/results/',
                        help='Path to save transformation file.')
    parser.add_argument('--max_corr_distance', type=float, default=1.5,
                        help='Maximum correspondence points-pair distance.')
    parser.add_argument('--downsampled_voxel_size', type=float, default=0.1,
                        help='downsampled voxel size for roughly global registration.')
    parser.add_argument('--factor', type=float, default=8,
                        help='downsampled factor for interpolating displacement to save computation.')

    args = parser.parse_args()

    run_pipeline(ref_img_path=args.ref_img_path, ref_pcd_path=args.ref_pcd_path, ref_meta_file=args.ref_meta_file,
                 sub_img_path=args.sub_img_path, sub_pcd_path=args.sub_pcd_path, sub_meta_path=args.sub_meta_path,
                 result_path=args.result_path, max_corr_distance=args.max_corr_distance * args.downsampled_voxel_size,
                 downsampled_voxel_size=args.downsampled_voxel_size, factor=args.factor)
