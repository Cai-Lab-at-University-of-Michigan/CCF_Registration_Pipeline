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
from memory_profiler import profile


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


def apply_transforms_torchV2(target_path, mv_path, aff_file, target_shape=None, what_dtype=np.uint8):
    def get_target_shape(target_path):
        target = read_nii_bysitk(target_path)
        target = sitk.GetArrayFromImage(target)
        target_shape = target.shape

        del target

        return target_shape

    if target_shape is None:
        target_shape = get_target_shape(target_path)

    aff = np.loadtxt(aff_file, dtype=np.float32)

    # get rid of translation part for re-centering
    aff[:3, 3] = 0

    aff = torch.from_numpy(aff[:3, :])
    aff = aff[None, ...]

    moving = read_nii_bysitk(mv_path)
    moving = sitk.GetArrayFromImage(moving)
    moving = np.array(moving, dtype=np.float32)

    # to tensor
    moving_min, moving_max = np.amin(moving), np.amax(moving)
    moving = moving / moving_max
    moving = torch.from_numpy(moving)
    moving = moving[None, None, ...]

    affined = warp_affine3d(moving, aff, dsize=target_shape)
    affined = affined.numpy()
    affined = np.squeeze(affined)
    affined = affined * moving_max

    affined = np.array(affined, dtype=what_dtype)

    return affined


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


@profile
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


def performAntsRegistrationV2(mv_path, target_path, registration_type='syn', record_path=None, tl_path=None, fname=None, outprefix=''):
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

    moving = ants.image_read(mv_path)
    moving = moving.numpy()
    moving_shape = moving.shape
    moving = rescale(moving, 1.0)
    moving = ants.from_numpy(moving)

    print(f'moving shape: {moving.shape}')

    target = ants.image_read(target_path)
    target = target.numpy()
    target = rescale(target, 1.0)
    target = ants.from_numpy(target)
    print(f'target shape: {target.shape}')

    start = time.time()
    if os.path.isfile(os.path.join(record_path, fname + '_affine.mat')) and os.path.isfile(os.path.join(record_path, fname + '_invdisp.nii.gz')):
        invtransforms = [os.path.join(record_path, fname + '_affine.mat'), os.path.join(record_path, fname + '_invdisp.nii.gz')]

        img_np = ants.apply_transforms(fixed=moving, moving=target,
                                         transformlist=invtransforms,
                                         interpolator='linear', verbose=True)

        img_np = img_np.numpy()
        img_np = np.transpose(img_np, (2, 1, 0))
        img_np = np.array(img_np, dtype=np.uint16)
        write_nii_bysitk(os.path.join(record_path, 'deformed', fname + '_warped_ccf.nii.gz'), img_np)

        if os.path.isfile(os.path.join(record_path, fname + '_disp.nii.gz')):
            transforms = [os.path.join(record_path, fname + '_disp.nii.gz'), os.path.join(record_path, fname + '_affine.mat')]

            img_np = ants.apply_transforms(fixed=target, moving=moving,
                                           transformlist=transforms,
                                           interpolator='linear', verbose=True)

            img_np = img_np.numpy()
            img_np = np.transpose(img_np, (2, 1, 0))
            img_np = np.array(img_np, dtype=np.uint8)
            write_nii_bysitk(os.path.join(record_path, 'deformed', fname + '_warped.nii.gz'), img_np)

    else:
        syn_res = ants.registration(fixed=target, moving=moving, type_of_transform='SyNCC',
                                    grad_step=0.2,
                                    flow_sigma=3,  # intra 3
                                    total_sigma=0.1,
                                    outprefix=outprefix,
                                    aff_metric='mattes',
                                    aff_sampling=32,
                                    syn_metric='mattes',
                                    syn_sampling=4)

        cmd = 'mv ' + syn_res['fwdtransforms'][0] + ' ' + os.path.join(record_path, fname + '_disp.nii.gz')
        cmd += '\n mv ' + syn_res['fwdtransforms'][1] + ' ' + os.path.join(record_path, fname + '_affine.mat')
        cmd += '\n mv ' + syn_res['invtransforms'][1] + ' ' + os.path.join(record_path, fname + '_invdisp.nii.gz')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        img_np = np.transpose(syn_res['warpedfixout'].numpy(), (2, 1, 0))
        img_np = np.array(img_np, dtype=np.uint16)
        write_nii_bysitk(os.path.join(record_path, 'deformed', fname + '_warped_ccf.nii.gz'), img_np)

        img_np = np.transpose(syn_res['warpedmovout'].numpy(), (2, 1, 0))
        img_np = np.array(img_np, dtype=np.uint8)
        write_nii_bysitk(os.path.join(record_path, 'deformed', fname + '_warped.nii.gz'), img_np)

    print('syn registration finished and takes: :', time.time() - start)

    # invtransforms = [os.path.join(record_path, fname + '_affine.mat'), os.path.join(record_path, fname + '_invdisp.nii.gz')]
    #
    # return invtransforms, moving_shape


def applyToAnnotation(invtransforms, record_path, moving_shape=None, tl_path=None, fname=None):
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

    tl_np = sitk.GetArrayFromImage(sitk.ReadImage(tl_path))
    l_target = ants.from_numpy(np.transpose(tl_np))

    if moving_shape is None:
        l_moving = l_target.clone()
    else:
        l_moving = np.ones(moving_shape)
        l_moving = np.array(l_moving, dtype=tl_np.dtype)
        l_moving = ants.from_numpy(l_moving)

    start = time.time()
    l_moving = ants.apply_transforms(fixed=l_moving, moving=l_target,
                                    transformlist=invtransforms,
                                    interpolator='nearestNeighbor', verbose=True)


    l_moving = l_moving.numpy()
    l_moving = np.transpose(l_moving, (2, 1, 0))
    l_moving = np.array(l_moving, dtype=np.uint32)
    write_nii_bysitk(os.path.join(record_path, 'deformed', fname + '_warped_ccf_annotation.nii.gz'), l_moving)

    print(f'Apply to annotation finished and takes {time.time() - start} s' )


@profile
def main():
    from img_utils import write_nii_bysitk, read_nii_bysitk
    from skimage.transform import resize
    from registration_utils import warp_affine3d
    import torch
    import numpy as np
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

    ccf = '/home/binduan/Downloads/NIH/tmp_test1/average_template_10.nii.gz'
    ccf_ann = '/home/binduan/Downloads/NIH/tmp_test1/annotation_10.nii.gz'
    moving = '/home/binduan/Downloads/NIH/tmp_test1/subject/ants/subject_affined/192341_red_mm_SLA.nii.gz'

    name = 'annotation_10'

    record_path = '/home/binduan/Downloads/NIH/tmp_test1/'

    maybe_mkdir_p(os.path.join(record_path, 'deformed'))

    invtransforms, moving_shape = performAntsRegistrationV2(
        mv_path=moving,
        target_path=ccf,
        tl_path=ccf_ann,
        registration_type='syn',
        record_path=record_path, fname=name,
        outprefix=join(record_path, 'temp'))

    applyToAnnotation(invtransforms, record_path, moving_shape, tl_path=ccf_ann, fname=name)


if __name__ == '__main__':
    import time
    from batchgenerators.utilities.file_and_folder_operations import *
    start_time = time.time()
    main()
    print(f'Time took: {time.time() - start_time} s')
