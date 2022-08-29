import itk
from utils.img_utils import write_nii_bysitk
import os
import numpy as np


def performElatixRegistration(ref_img_path, sub_img_path, ref_ann_path, sub_save_path, name):
    moving_image = itk.imread(ref_img_path, itk.F)
    fixed_image = itk.imread(sub_img_path, itk.F)

    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('affine')
    parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')

    parameter_object.AddParameterMap(parameter_map_rigid)
    parameter_object.AddParameterMap(parameter_map_bspline)

    # Call registration function and specify output directory
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        output_directory=sub_save_path)


    write_nii_bysitk(os.path.join(sub_save_path, name + '_warped_ccf.nii.gz'), np.asarray(result_image).astype(np.float32))

    moving_image = itk.imread(ref_ann_path, itk.F)

    # Change interpolation type to nearestNeighbor
    result_transform_parameters.SetParameter(0, 'FinalBSplineInterpolationOrder', '0')
    result_transform_parameters.SetParameter(1, 'FinalBSplineInterpolationOrder', '0')

    # Load Transformix Object
    transformix_object = itk.TransformixFilter.New(moving_image)
    transformix_object.LogToConsoleOn()
    transformix_object.SetTransformParameterObject(result_transform_parameters)

    # Update object (required)
    transformix_object.UpdateLargestPossibleRegion()

    # Results of Transformation
    result_image = transformix_object.GetOutput()
    write_nii_bysitk(os.path.join(sub_save_path, name +  '_warped_ccf_annotation.nii.gz'), np.asarray(result_image).astype(np.float32))
