from collections import OrderedDict

import neuroglancer
import numpy as np


# COLORS = "byrgmkc"


# def convert_contour_lines(contours_list, axis=0):
#     def list_of_coordinates(coords, slice, axis):
#         if axis == 0:
#             return list([slice, coords[0], coords[1]])
#         elif axis == 1:
#             return list([coords[0], slice, coords[1]])
#         else:
#             return list([coords[0], coords[1], slice])
#
#     # colors = []
#     lines_list = []
#     for i, each_countours in enumerate(contours_list):
#         for n, contour in enumerate(each_countours):
#             # if n < len(COLORS):
#             #     c1 = COLORS[n]
#             # else:
#             #     c1 = np.random.rand(1,3)
#             # colors.append(colorSet.to_hex(c1))
#             lines = []
#             for nn, (pt1, pt2) in enumerate(zip(contour[:-1], contour[1:])):
#                 line = neuroglancer.LineAnnotation()
#                 line.pointA = list_of_coordinates(pt1, i, axis)
#                 line.pointB = list_of_coordinates(pt2, i, axis)
#                 line.id = 'contour-' + str(i) + '-' + str(nn)
#                 lines.append(line)
#             lines_list.append(lines)
#
#     return lines_list


# def contours_to_stack(contours_list, axis, shape_):
#     img = np.zeros(shape_, dtype=np.uint8)
#     for i, each_countours in enumerate(contours_list):
#         for ii, contour in enumerate(each_countours):
#             rep_i = np.asarray(i).repeat(len(contour))
#             contour = (contour).astype(np.int64)
#             if axis == 0:
#                 img[rep_i, contour[:, 0], contour[:, 1]] = 255
#             elif axis == 1:
#                 img[contour[:, 0], rep_i, contour[:, 1]] = 255
#             else:
#                 img[contour[:, 0], contour[:, 1], rep_i] = 255
#
#     return img


def coords_to_stack(coors_list, shape_, canonical=True):
    """coordinates order be canonical XYZ or ZYX"""
    img = np.zeros(shape_, dtype=np.uint8)
    if canonical:
        img[coors_list[:, 2], coors_list[:, 1], coors_list[:, 0]] = 255 # since image is ZYX
    else:
        img[coors_list[:, 0], coors_list[:, 1], coors_list[:, 2]] = 255

    return img


def view_in_neuroglancer(odict: OrderedDict, scales=[1, 1, 1], units='mm',
                         names=['coronal', 'sagittal', 'horizontal']):
    dimensions = neuroglancer.CoordinateSpace(
        scales=scales,
        units=units,
        names=names)

    viewer = neuroglancer.Viewer()
    print(viewer)
    with viewer.txn() as s:
        for key, value in odict.items():
            if '_gray' in key:
                shape_ = value.shape
                s.layers[key] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(
                        value / np.amax(value),
                        dimensions=dimensions),
                    shader="""
            void main() {
              emitGrayscale(normalized());
            """,
                )
            elif '_fg' in key:
                s.layers[key] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        value,
                        dimensions=dimensions,
                    ))

            elif '_contour' in key:
                s.layers[key] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        coords_to_stack(value, shape_=shape_),
                        dimensions=dimensions),
                )
            else:
                print(f'key should contain string [_gray | _seg], current key is {key}')
                continue

    return viewer


if __name__ == '__main__':
    from batchgenerators.utilities.file_and_folder_operations import *
    from img_utils import read_nii_bysitk
    import SimpleITK as sitk

    img_obj, img_info = read_nii_bysitk(
        '/home/duanbin/Downloads/NIH/download.brainlib.org/hackathon/2022_GYBS/data/reference/average_template_25_mm_ASL.nii.gz',
        metadata=True)

    name = 'average_template_25_mm_ASL'

    # visulize in neuroglancer
    fg_img_obj = read_nii_bysitk(
        '/home/duanbin/Downloads/NIH/hackathon/results/reference/' + 'mask_' + name + '.nii.gz', metadata=False)

    fg_contour = load_pickle('/home/duanbin/Downloads/NIH/hackathon/results/reference/' + 'contour_' + name + '.pkl')

    view_in_neuroglancer(OrderedDict({'ref_gray': sitk.GetArrayFromImage(img_obj),
                                      'ref_fg': sitk.GetArrayFromImage(fg_img_obj),
                                      'ref_contour': fg_contour}),
                         scales=img_info['spacing'],
                         units='mm', names=['sagittal', 'horizontal', 'coronal'])
