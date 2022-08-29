import SimpleITK as sitk
import numpy as np
from tifffile import TiffFile
import os


def get_physical_spacing(file_str):
    """convert to mm"""
    img_obj = sitk.ReadImage(file_str)
    spacing = img_obj.GetSpacing()
    return np.array(spacing)


def convert_to_nii(input, output, what_dtype=np.uint8, normalize=False):
    if not os.path.isfile(output):
        with TiffFile(input) as tif:
            img_np = tif.asarray()
            if normalize:
                img_np = img_np / np.amax(img_np) * 255

            img_np = np.array(img_np, dtype=what_dtype)
            imagej_metadata = tif.imagej_metadata
        img_obj = sitk.GetImageFromArray(img_np)
        z_spacing = imagej_metadata['spacing']
        x_spacing = convert_spacing(tif.pages[0].tags['XResolution'].value)
        y_spacing = convert_spacing(tif.pages[0].tags['YResolution'].value)
        spacing = np.array([x_spacing, y_spacing, z_spacing])
        if imagej_metadata['unit'] == 'micron':
            spacing = spacing / 1000
        img_obj.SetSpacing(spacing)
        sitk.WriteImage(img_obj, output)
    else:
        spacing = get_physical_spacing(output)

    return output, spacing


def convert_spacing(v: tuple):
    return 1 / v[0] * v[1]


def read_nii_bysitk(file_str, metadata=False):
    """ read nii.gz to numpy through simpleitk
    """
    img_obj = sitk.ReadImage(file_str)
    if metadata:
        info_obj = {
            "spacing": img_obj.GetSpacing(),
            "origin": img_obj.GetOrigin(),
            "direction": img_obj.GetDirection(),
        }
        return img_obj, info_obj
    else:
        return img_obj


def write_nii_bysitk(dst_file_str, img_np, src_obj=None, spacing=None):
    img_obj = sitk.GetImageFromArray(img_np)
    if src_obj is not None:
        img_obj.CopyInformation(src_obj)

    if spacing is not None:
        img_obj.SetSpacing(spacing)

    # if src_obj is None:
    #     if img_np.ndim == 4: # displacement
    #         img_obj.Se
    #     elif img_np.ndim == 3: # image
    #         pass
    #     else:
    #         raise RuntimeError(f"Image should 3D or 3D flow information (4D), currently is {img_np.ndim} ")

    sitk.WriteImage(img_obj, dst_file_str)


def normalize_2d(img):
    """normalize using max value in the image"""
    if img.ndim > 2:
        raise RuntimeError('image should be 2d grayscale')

    return img / np.amax(img)
