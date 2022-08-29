import argparse
import subprocess
import time

import numpy as np
import scipy
from batchgenerators.utilities.file_and_folder_operations import *

from utils.ants_utils import performAntsRegistrationV2, apply_transforms_torchV2, applyToAnnotation
from utils.img_utils import write_nii_bysitk, convert_to_nii, get_physical_spacing
from utils.registration_utils import compose_affine_transforms
from utils.elastix_utils import performElatixRegistration
import SimpleITK as sitk
from utils import print_options


def run_pipeline(ref_img_path, ref_ann_path, sub_img_path, result_path, suffix):
    ref_save_path = join(result_path, 'reference')
    sub_save_path = join(result_path, 'subject')

    maybe_mkdir_p(ref_save_path)
    maybe_mkdir_p(sub_save_path)

    start_time = time.time()
    print('Starting...')

    if ref_img_path.endswith('.tif'):
        file_str = join(result_path, ref_img_path.split('/')[-1].replace('.tif', '.nii.gz'))
        ref_img_path, ref_img_spacing = convert_to_nii(ref_img_path, file_str, normalize=True)
    else:
        ref_img_spacing = get_physical_spacing(ref_img_path)

    if ref_ann_path.endswith('.tif'):
        file_str = join(result_path, ref_ann_path.split('/')[-1].replace('.tif', '.nii.gz'))
        ref_ann_path, ref_ann_spacing = convert_to_nii(ref_ann_path, file_str, what_dtype=np.float32)
    else:
        ref_ann_spacing = get_physical_spacing(ref_ann_path)

    if sub_img_path.endswith('.tif'):
        file_str = join(result_path, sub_img_path.split('/')[-1].replace('.tif', '.nii.gz'))
        sub_img_path, sub_img_spacing = convert_to_nii(sub_img_path, file_str)
    else:
        sub_img_spacing = get_physical_spacing(sub_img_path)

    print(f'files conversion took {time.time() - start_time} s, accumulated time: {time.time() - start_time} s')
    step_time = time.time()

    ###step 1:###
    #    1. segment reference image if not yet
    ref_name = ref_img_path.split('/')[-1].split('.nii.gz')[0]
    if not os.path.isfile(join(ref_save_path, 'point_cloud', ref_name + '.pcd')):
        cmd = f'python reference_pipeline.py --ref_path {ref_img_path} --result_path {ref_save_path}'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        print(f'reference segmentation took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')
        step_time = time.time()

    ref_meta_file = join(ref_save_path, 'reference.meta')

    name = sub_img_path.split('/')[-1].split('.nii.gz')[0]

    ###step 2:###
    #    1. downsample subject to ccf to the same physical spacing
    #    2. preprocess to remove bright stripes
    if not os.path.isfile(join(sub_save_path, 'preprocessed', name + '.nii.gz')):
        cmd = f'python preprocess_subject.py --sub_path {sub_img_path} --result_path {sub_save_path} --ref_meta_file {ref_meta_file}'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        print(f'preprocess subject took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')
        step_time = time.time()

    ###step 3:###
    #    2. segment subject image if not yet
    if not os.path.isfile(join(sub_save_path, 'point_cloud', name + '.pcd')):
        preprocessed_sub_path = join(sub_save_path, 'preprocessed', name + '.nii.gz')
        cmd = f'python subject_pipeline.py --sub_path {preprocessed_sub_path} --result_path {sub_save_path} --ref_path {ref_save_path} --ref_meta_file {ref_meta_file}'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        print(f'subject segmentation took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')
        step_time = time.time()

    ###step 4:###
    #    1. point cloud global registration for the subject to ccf
    if not os.path.isfile(join(sub_save_path, 'warped_image_affine', name + '.nii.gz')):
        red_pcd_path = subfiles(join(ref_save_path, 'point_cloud'), prefix=ref_name, suffix='.pcd')[0]
        sub_pcd_path = subfiles(join(sub_save_path, 'point_cloud'), prefix=name, suffix='.pcd')[0]
        sub_meta_path = subfiles(join(sub_save_path, 'meta'), prefix=name, suffix='.meta')[0]

        prealigned_sub_path = join(sub_save_path, 'prealigned', name + '.nii.gz')

        cmd = f'python registration_pipeline.py --ref_img_path {ref_img_path} --ref_pcd_path {red_pcd_path} --ref_meta_file {ref_meta_file}'
        cmd += f' --sub_img_path {prealigned_sub_path} --sub_pcd_path {sub_pcd_path} --sub_meta_path {sub_meta_path} --result_path {sub_save_path}'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        print(f'point cloud and deformation took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')
        step_time = time.time()

    ###step 5:###
    #   1. compose affines from different stage
    #   2. calculate inverse transform
    aff = []
    init_aff = np.eye(4)
    # init_aff[:3, :3] = 0.8 * init_aff[:3, :3]
    aff.append(init_aff)
    aff.append(np.loadtxt(join(sub_save_path, 'affines', name + '_pre_scaling.txt')))
    aff.append(np.loadtxt(join(sub_save_path, 'affines', name + '_cortex.txt')))
    aff.append(np.loadtxt(join(sub_save_path, 'affines', name + '_pcd.txt')))

    forward_aff = compose_affine_transforms(aff)

    np.savetxt(join(sub_save_path, 'affines', name + '_sub2ccf_wo.txt'), forward_aff)

    backward_aff = scipy.linalg.inv(forward_aff)
    np.savetxt(join(sub_save_path, 'affines', name + '_ccf2sub_wo.txt'), backward_aff)

    print(f'forward_aff: {forward_aff}')
    print(f'backward_aff: {backward_aff}')

    ###step 6:###
    #    1. non-linear deformation to ccf
    # using elastix

    if not os.path.isfile(join(sub_save_path, 'elastix', 'ccf_affined', name + '_warped_ccf_annotation' + suffix)):
        print(f'using elastix to register sample: {sub_img_path.split("/")[-1]}')

        record_path = join(sub_save_path, 'elastix')
        maybe_mkdir_p(join(record_path, 'subject_affined'))
        maybe_mkdir_p(join(record_path, 'ccf_affined'))
        maybe_mkdir_p(join(record_path, 'deformed'))

        # apply affine to raw image
        if not os.path.isfile(join(record_path, 'subject_affined', name + '.nii.gz')):
            affined_raw = apply_transforms_torchV2(target_path=ref_img_path, mv_path=sub_img_path,
                                                   aff_file=join(sub_save_path, 'affines', name + '_sub2ccf_wo.txt'), what_dtype=np.uint8)

            write_nii_bysitk(join(record_path, 'subject_affined', name + '.nii.gz'), affined_raw, spacing=ref_img_spacing)
            del affined_raw

            print(f'Subject affine transformation took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')
            step_time = time.time()

        # non-linear deformation
        if not os.path.isfile(join(record_path, 'deformed', name + '_warped_ccf.nii.gz')):
            performElatixRegistration(ref_img_path, join(record_path, 'subject_affined', name + '.nii.gz'), ref_ann_path, join(record_path, 'deformed'), name=name)

            print(f'Non-linear deformation took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')
            step_time = time.time()

        # apply affine to match size
        if not os.path.isfile(join(record_path, 'ccf_affined', name + '_warped_ccf_annotation' + suffix)):
            affined = apply_transforms_torchV2(target_path=sub_img_path, mv_path=join(record_path, 'deformed', name + '_warped_ccf.nii.gz'),
                                               aff_file=join(sub_save_path, 'affines', name + '_ccf2sub_wo.txt'), what_dtype=np.uint16)

            affined = sitk.GetImageFromArray(affined)
            affined.SetSpacing(sub_img_spacing)
            sitk.WriteImage(affined, join(record_path, 'ccf_affined', name + '_warped_ccf' + suffix))

            affined = apply_transforms_torchV2(target_path=sub_img_path, mv_path=join(record_path, 'deformed', name + '_warped_ccf_annotation.nii.gz'),
                                               aff_file=join(sub_save_path, 'affines', name + '_ccf2sub_wo.txt'), what_dtype=np.float32)

            affined = sitk.GetImageFromArray(affined)
            affined.SetSpacing(sub_img_spacing)
            sitk.WriteImage(affined, join(record_path, 'ccf_affined', name + '_warped_ccf_annotation' + suffix))

            print(f'Apply affine to match size took {time.time() - step_time} s, accumulated time: {time.time() - start_time} s')

    print(f'Total time: {time.time() - start_time} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reference pipeline for 2022 Brain registration hackathon.')
    parser.add_argument('--ref_img_path', type=str,
                        default='/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/reference/average_template_10.tif',
                        help='Path to the ccf image file (nii.gz).')
    parser.add_argument('--ref_ann_path', type=str,
                        default='/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/reference/annotation_10.tif',
                        help='Path to the ccf annotation file (nii.gz).')
    parser.add_argument('--sub_img_path', type=str,
                        default='/home/binduan/Downloads/NIH/192341_red_mm_SLA.tif',
                        help='Path to the raw image file (.nii.gz).')
    parser.add_argument('--result_path', type=str,
                        default='/home/binduan/Downloads/NIH/tmp_test3/',
                        help='Path to intermediate results.')
    parser.add_argument('--suffix', type=str, default='.tif',
                        help='Which file format to be used for saving final result.')

    args = parser.parse_args()
    print_options(parser, args)

    run_pipeline(ref_img_path=args.ref_img_path, ref_ann_path=args.ref_ann_path, sub_img_path=args.sub_img_path,
                 result_path=args.result_path, suffix=args.suffix)
