import copy
import os
import subprocess
import time

import ants
import numpy as np
from skimage.transform import rescale
import SimpleITK as sitk
from utils.registration_utils import *
from utils.img_utils import *


def apply_transforms(target_path, mv_path, transformlist, verbose, ml_path=None):
    loutput = None
    moving = ants.image_read(mv_path)
    moving = moving.numpy()

    moving = rescale(moving, 1.0)
    moving = ants.from_numpy(moving)

    target = ants.image_read(target_path)
    target = target.numpy()
    target_shape = target.shape

    target = rescale(target, 1.0)
    target = ants.from_numpy(target)

    if ml_path is not None:
        ml_sitk = sitk.ReadImage(ml_path)
        ml_np = sitk.GetArrayFromImage(ml_sitk)
        l_moving = ants.from_numpy(np.transpose(ml_np))

        l_target = np.ones(target_shape)
        l_target = np.array(l_target, dtype=ml_np.dtype)
        l_target = ants.from_numpy(l_target)

    affined = ants.apply_transforms(fixed=target, moving=moving,
                          transformlist=transformlist,
                          interpolator='linear',
                          verbose=verbose)


    if ml_path is not None:
        loutput = ants.apply_transforms(fixed=l_target, moving=l_moving, transformlist=transformlist, interpolator='nearestNeighbor')
        loutput = loutput.numpy()

    affined = affined.numpy()

    affined = np.transpose(affined, (2, 1, 0))
    loutput = np.transpose(loutput, (2, 1, 0)) if loutput is not None else None

    return affined, loutput


def apply_transforms_torch(target_path, mv_path, aff_file, ml_path=None):
    aff = np.loadtxt(aff_file)

    # get rid of translation part for re-centering
    aff[:3, 3] = 0

    aff = torch.from_numpy(aff[:3, :])
    aff = aff[None, ...]

    loutput = None
    moving = read_nii_bysitk(mv_path)
    moving = sitk.GetArrayFromImage(moving)

    # to tensor
    moving_min, moving_max = np.amin(moving), np.amax(moving)
    moving = moving / moving_max
    moving = torch.from_numpy(moving)
    moving = moving[None, None, ...]

    target = read_nii_bysitk(target_path)
    target = sitk.GetArrayFromImage(target)
    target_shape = target.shape

    # to tensor
    # target_min, target_max = np.amin(target), np.amax(target)
    # target = target / target_max
    # target = torch.from_numpy(target)
    # target = target[None, None, ...]

    if ml_path is not None:
        ml = read_nii_bysitk(ml_path)
        ml = sitk.GetArrayFromImage(ml)

        # to tensor
        ml_min, ml_max = np.amin(ml), np.amax(ml)
        ml = ml / ml_max
        ml = torch.from_numpy(ml)
        ml = ml[None, None, ...]


    affined = warp_affine3d(moving, aff, dsize=target_shape)
    affined = affined.numpy()
    affined = np.squeeze(affined)
    affined = affined * moving_max

    if ml_path is not None:
        loutput = warp_affine3d(ml, aff, dsize=target_shape, flags='nearest')
        loutput = loutput.numpy()
        loutput = np.squeeze(loutput)
        loutput = loutput * ml_max

    return affined, loutput


def expand_batch_ch_dim(input):
    """
    expand dimension [1,1] +[x,y,z]

    :param input: numpy array
    :return:
    """
    if input is not None:
        return np.expand_dims(np.expand_dims(input, 0), 0)
    else:
        return None


def performAntsRegistration(mv_path, target_path, registration_type='syn', record_path=None, tl_path=None, fname=None, outprefix=''):
    """
    call [AntsPy](https://github.com/ANTsX/ANTsPy),
    :param mv_path: path of moving image
    :param target_path: path of target image
    :param registration_type: type of registration, support 'affine' and 'syn'(include affine)
    :param record_path: path of saving results
    :param tl_path: path of label of target image
    :param fname: pair name or saving name of the image pair
    :return: warped image (from moving to target), warped label (from target to moving)
    """
    loutput = None

    moving = ants.image_read(mv_path)
    moving = moving.numpy()
    moving_shape = moving.shape

    moving = rescale(moving, 1.0)
    moving = ants.from_numpy(moving)

    target = ants.image_read(target_path)
    target = target.numpy()
    target = rescale(target, 1.0)
    target = ants.from_numpy(target)

    if tl_path is not None:
        tl_sitk = sitk.ReadImage(tl_path)
        tl_np = sitk.GetArrayFromImage(tl_sitk)
        # tl_np = rescale(tl_np, 1.0, order=0)
        # write_nii_bysitk('/home/binduan/Downloads/NIH/tmp_test/ccf_to_sub/test.nii.gz', tl_np)
        l_target = ants.from_numpy(np.transpose(tl_np))

        l_moving = np.ones(moving_shape)
        l_moving = np.array(l_moving, dtype=tl_np.dtype)
        l_moving = ants.from_numpy(l_moving)

        # l_target = ants.from_numpy(np.transpose(tl_np), spacing=target.spacing, direction=target.direction,
        #                            origin=target.origin)

    start = time.time()
    # print("param_in_ants:{}".format(param_in_ants))
    if registration_type == 'affine':
        affine_file = ants.affine_initializer(moving, target)
        af_img = ants.apply_transforms(fixed=moving, moving=target, transformlist=affine_file)
        if tl_path is not None:
            loutput = ants.apply_transforms(fixed=l_moving, moving=l_target, transformlist=affine_file,
                                            interpolator='nearestNeighbor')
            loutput = loutput.numpy()
        warpedfixout = af_img.numpy()

        affine_file = ants.affine_initializer(target, moving)
        af_img = ants.apply_transforms(fixed=target, moving=moving, transformlist=affine_file)
        warpedmovout = af_img.numpy()
        print('affine registration finished and takes: :', time.time() - start)

    if registration_type == 'syn':
        syn_res = ants.registration(fixed=target, moving=moving, type_of_transform='SyNCC',
                                    grad_step=0.2,
                                    flow_sigma=3,  # intra 3
                                    total_sigma=0.1,
                                    outprefix=outprefix,
                                    aff_metric='mattes',
                                    aff_sampling=8,
                                    syn_metric='mattes',
                                    syn_sampling=32,
                                    reg_iterations=(20, 10, 0),
                                    aff_iterations=(210, 120, 120, 10))

        if tl_path is not None:
            time.sleep(1)
            loutput = ants.apply_transforms(fixed=l_moving, moving=l_target,
                                            transformlist=syn_res['invtransforms'],
                                            interpolator='nearestNeighbor', verbose=True)
            loutput = loutput.numpy()

        warpedfixout = syn_res['warpedfixout'].numpy()
        warpedmovout = syn_res['warpedmovout'].numpy()
        print('syn registration finished and takes: :', time.time() - start)

        cmd = 'mv ' + syn_res['fwdtransforms'][1] + ' ' + os.path.join(record_path, fname + '_disp.nii.gz')
        cmd += '\n mv ' + syn_res['fwdtransforms'][0] + ' ' + os.path.join(record_path, fname + '_affine.mat')
        cmd += '\n mv ' + syn_res['invtransforms'][0] + ' ' + os.path.join(record_path, fname + '_invdisp.nii.gz')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    warpedfixout = np.transpose(warpedfixout, (2, 1, 0))
    warpedmovout = np.transpose(warpedmovout, (2, 1, 0))

    loutput = np.transpose(loutput, (2, 1, 0)) if loutput is not None else None

    return warpedfixout, warpedmovout, loutput


# if __name__ == '__main__':
#     from batchgenerators.utilities.file_and_folder_operations import *
#     from img_utils import write_nii_bysitk, read_nii_bysitk
#     from skimage.transform import resize
#     from registration_utils import warp_affine3d
#     import torch
#     import numpy as np
#
#     ccf = '/home/binduan/Downloads/NIH/tmp_test/ccf_to_sub/average_template_25_warped.nii.gz affined.tif'
#     ccf_ann = '/home/binduan/Downloads/NIH/tmp_test/ccf_to_sub/annotation_25_warped.nii.gz affined.tif'
#     moving = '/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/subject/192341_red_mm_SLA.nii.gz'
#
#     record_path = '/home/binduan/Downloads/NIH/tmp_test/ccf_to_sub_final/'
#
#     # warp ccf_ann to moving image
#     output, loutput = performAntsRegistration(mv_path=moving, target_path=ccf,
#                                               tl_path=ccf_ann,
#                                               registration_type='affine',
#                                               record_path=record_path, fname='annotation_25',
#                                               outprefix=join(record_path, 'temp'))
#
#     output = resize(output, output_shape=(1128, 1029, 590), order=1)
#     loutput = resize(loutput, output_shape=(1128, 1029, 590), order=0)
#
#     print(output.shape, loutput.shape)
#     print(np.amin(output), np.amax(output))
#     print(np.amin(loutput), np.amax(loutput))
#
#     output = np.array(output, dtype=np.uint16)
#     loutput = np.array(loutput, dtype=np.uint32)
#
#     write_nii_bysitk(join(record_path, 'average_template_25_warped.nii.gz'), output)
#     write_nii_bysitk(join(record_path, 'annotation_25_warped.nii.gz'), loutput)

    # ccf = '/home/binduan/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/reference/average_template_25_mm_ASL.nii.gz'
    # ccf_ann = '/home/binduan/Downloads/NIH/hackathon/results/reference/annotation_25.nii.gz'
    # moving = '/home/binduan/Downloads/NIH/hackathon/results/subject/warped_image_affine/192341_red_mm_SLA.nii.gz'
    # aff = '/home/binduan/Downloads/NIH/tmp_test/subject/affines/192341_red_mm_SLA_ccf2sub_wo.txt'
    #
    # record_path = '/home/binduan/Downloads/NIH/tmp_test/ccf_to_sub/'
    #
    # # warp ccf_ann to moving image
    # output, loutput = performAntsRegistration(mv_path=moving, target_path=ccf,
    #                                           tl_path=ccf_ann,
    #                                           registration_type='syn',
    #                                           record_path=record_path, fname='annotation_25',
    #                                           outprefix=join(record_path, 'temp'))
    #
    # output = resize(output, output_shape=(456, 320, 528), order=1)
    # loutput = resize(loutput, output_shape=(456, 320, 528), order=0)
    #
    # print(output.shape, loutput.shape)
    # print(np.amin(output), np.amax(output))
    # print(np.amin(loutput), np.amax(loutput))
    #
    # output = np.array(output, dtype=np.uint16)
    # loutput = np.array(loutput, dtype=np.uint32)
    #
    # write_nii_bysitk(join(record_path, 'average_template_25_warped.nii.gz'), output)
    # write_nii_bysitk(join(record_path, 'annotation_25_warped.nii.gz'), loutput)



