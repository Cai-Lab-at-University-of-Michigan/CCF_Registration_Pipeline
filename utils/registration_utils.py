from __future__ import print_function

import copy
import itertools
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import *
from kornia.geometry.conversions import (
    convert_affinematrix_to_homography3d,
    normalize_homography3d,
)
from kornia.utils.helpers import _torch_inverse_cast
from probreg import cpd, l2dist_regs, bcpd
from probreg.transformation import Transformation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.spatial import distance


######################################
####### Transformation Utils #########
######################################
def calculate_warped_shape(shape_, aff):
    coords = []
    for z in [-shape_[0] // 2, shape_[0] // 2 + 1]:
        for y in [-shape_[1] // 2, shape_[1] // 2 + 1]:
            for x in [-shape_[2] // 2, shape_[2] // 2 + 1]:
                coords.append([x, y, z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coords))

    pcd.transform(aff)

    coords = np.array(pcd.points)

    xx = int(math.ceil(np.amax(coords[:, 0]) - np.amin(coords[:, 0])))
    yy = int(math.ceil(np.amax(coords[:, 1]) - np.amin(coords[:, 1])))
    zz = int(math.ceil(np.amax(coords[:, 2]) - np.amin(coords[:, 2])))

    return (zz, yy, xx)


def get_voxel_scaling_matrix(spacing):
    trans = np.eye(4)
    trans[0, 0] = spacing[0]
    trans[1, 1] = spacing[1]
    trans[2, 2] = spacing[2]

    return trans


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def get_downsampled_shape(image_shape, factor: int = 8):
    w, h, d = image_shape  # ZYX

    return (w // factor, h // factor, d // factor)


def compose_affine_transforms(affine_list: List):
    aff = np.eye(4)

    for a in reversed(affine_list):
        aff = aff @ a

    return aff


def warp_affine3d(
        src: torch.Tensor,
        M: torch.Tensor,
        dsize: Tuple[int, int, int],
        flags: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = True,
        normal: bool = False
) -> torch.Tensor:
    r"""Apply a projective transformation a to 3d tensor.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Args:
        src : input tensor of shape :math:`(B, C, D, H, W)`.
        M: projective transformation matrix of shape :math:`(B, 3, 4)`.
        dsize: size of the output image (depth, height, width).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners : mode for grid_generation.

    Returns:
        torch.Tensor: the warped 3d tensor with shape :math:`(B, C, D, H, W)`.

    .. note::
        This function is often used in conjunction with :func:`get_perspective_transform3d`.
    """
    if len(src.shape) != 5:
        raise AssertionError(src.shape)
    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 4)):
        raise AssertionError(M.shape)
    if len(dsize) != 3:
        raise AssertionError(dsize)
    B, C, D, H, W = src.size()

    size_src: Tuple[int, int, int] = (D, H, W)
    size_out: Tuple[int, int, int] = dsize

    M_4x4 = convert_affinematrix_to_homography3d(M)  # Bx4x4

    # we need to normalize the transformation since grid sample needs -1/1 coordinates
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(M_4x4, size_src, size_out)  # Bx4x4

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    P_norm: torch.Tensor = src_norm_trans_dst_norm[:, :3]  # Bx3x4

    # compute meshgrid and apply to input
    dsize_out: List[int] = [B, C] + list(size_out)
    grid = torch.nn.functional.affine_grid(P_norm, dsize_out, align_corners=align_corners)

    # recenter
    for i in range(len(dsize)):
        max_grid, min_grid = torch.amax(grid[..., i]), torch.amin(grid[..., i])
        avg_grid = 0.5 * (max_grid + min_grid)
        grid[..., i] = grid[..., i] - avg_grid

    if normal:
        # resample to [-1, 1]
        for i in range(len(dsize)):
            max_grid, min_grid = torch.amax(grid[..., i]), torch.amin(grid[..., i])
            grid[..., i] = torch.where(grid[..., i] >= 0, grid[..., i] / max_grid, -grid[..., i] / min_grid)

    return torch.nn.functional.grid_sample(
        src, grid, align_corners=align_corners, mode=flags, padding_mode=padding_mode
    )


def prepare_affine_matrix(trans_sub, mid_aff, trans_ref):
    inv_trans_ref = np.eye(4)
    inv_trans_ref[np.diag_indices_from(trans_ref)] = 1 / trans_ref[np.diag_indices_from(trans_ref)]
    new_aff = [trans_sub]
    new_aff.extend(mid_aff)
    new_aff.append(inv_trans_ref)

    aff = compose_affine_transforms(new_aff)

    return aff


############################
####### BBox Utils  ########
############################
def box3d_iou(ref_bbox, sub_bbox):
    vol1 = ref_bbox.volume()
    vol2 = sub_bbox.volume()

    ref_min_bound = ref_bbox.min_bound
    sub_min_bound = sub_bbox.min_bound
    min_bound = np.where(ref_min_bound > sub_min_bound, ref_min_bound, sub_min_bound)

    ref_max_bound = ref_bbox.max_bound
    sub_max_bound = sub_bbox.max_bound
    max_bound = np.where(ref_max_bound < sub_max_bound, ref_max_bound, sub_max_bound)

    inter_xyz = np.where(max_bound - min_bound > 0, max_bound - min_bound, 0)

    inter_vol = np.prod(inter_xyz)

    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou


############################
####### Functions  #########
############################
def guess_cortex_position(long_axis_vs, dists, factor, visualized=False, title=None, trial=0):
    if title is not None:
        wo_trial_name = title.split('_trial_')[0]
        new_title = f'{wo_trial_name}_trial_{trial + 1}'
    else:
        new_title = None

    """Recursively guess cortex width"""
    sigma = factor / np.std(dists)

    global_max = np.where(dists == np.amax(dists))[0][0]
    filter_dists = gaussian_filter1d(dists, sigma=sigma)

    grad = np.gradient(filter_dists)
    grad -= np.amin(grad)
    grad /= np.amax(grad)

    # exclude the first and last 5% slices

    # local maximum of grad
    local = argrelextrema(grad, np.greater)[0]

    local_maxis = []
    for m in local:
        if int(0.01 * dists.shape[0]) < m < int(0.99 * dists.shape[0]):
            local_maxis.append(m)

    # local_maxis = np.array(local_maxis)
    #
    # if len(local_maxis) < 2:
    #     return guess_cortex_position(long_axis_vs, dists, factor=0.9 * factor, title=new_title, trial=trial+1)
    #
    # # pick the top 2 surrounding the maximum
    # starting_index = np.where(grad == np.amax(grad[local_maxis[local_maxis < global_max]]))[0][0]
    # ending_index = np.where(grad == np.amax(grad[local_maxis[local_maxis > global_max]]))[0][0]

    grad_grad = np.gradient(grad)
    grad_grad -= np.amin(grad_grad)
    grad_grad /= np.amax(grad_grad)

    # find where gradient changes the direction so that there is a valley or peak
    local = argrelextrema(grad_grad, np.greater)[0]

    local_zeros = []
    for m in local:
        if int(0.05 * dists.shape[0]) < m < int(0.95 * dists.shape[0]):
            local_zeros.append(m)

    local_zeros = np.array(local_zeros)

    if min(len(local_zeros), np.sum(local_zeros < global_max), np.sum(local_zeros > global_max)) < 2:
        print("reducing smoothing factor")
        return guess_cortex_position(long_axis_vs, dists, factor=0.8 * factor, title=new_title, trial=trial + 1)

    # starting_index = local_zeros[np.argmin(np.abs(local_zeros - starting_index))]
    # ending_index = local_zeros[np.argmin(np.abs(local_zeros - ending_index))]

    starting_index = np.where(grad_grad == np.amax(grad_grad[local_zeros[local_zeros < global_max]]))[0][0]
    ending_index = np.where(grad_grad == np.amax(grad_grad[local_zeros[local_zeros > global_max]]))[0][0]

    if np.mean(dists[starting_index:global_max]) <= np.mean(dists[global_max:ending_index]):
        ending_index = local_zeros[local_zeros > global_max][0]
    else:
        starting_index = local_zeros[local_zeros < global_max][-1]

    width = long_axis_vs[ending_index] - long_axis_vs[starting_index]

    if visualized or title is not None:
        plt.plot(dists, label='dist')
        plt.plot(filter_dists, label='filter_dist')
        # plt.plot(grad, label='grad')
        plt.plot(grad_grad, label='grad_grad')
        plt.axhline(y=0, ls='-', c='r')
        plt.legend()

        plt.scatter(global_max, np.amax(dists), marker='^', c='r')

        # plt.scatter(starting_index, grad[starting_index], marker='o', c='g')
        # plt.scatter(ending_index, grad[ending_index], marker='o', c='b')

        # plt.scatter(local_maxis, grad[local_maxis], marker='*', c='g')
        plt.scatter(local_zeros, grad_grad[local_zeros], marker='*')

        plt.axvline(x=starting_index, ls='-', c='b')
        plt.axvline(x=ending_index, ls='-', c='b')

        plt.scatter(starting_index, dists[starting_index], marker='o', c='r')
        plt.scatter(ending_index, dists[ending_index], marker='o', c='r')

        plt.title(title)

        plt.show()

    # expecting the cortex is at least this much of the whole slice
    if width > 0.4 * (long_axis_vs[-1] - long_axis_vs[0]):
        return starting_index, ending_index
    else:
        print("Increasing smoothing factor")
        return guess_cortex_position(long_axis_vs, dists, factor=1.2 * factor, title=new_title, trial=trial + 1)


def find_cortex_bbox(pcd, factor=4, return_indices=False, title=None):
    """return a bounding box around cortex (Return type: PointCloud)
    """
    # calculate all areas along the longest axis
    AXES = [0, 1, 2]
    points = np.array(pcd.points)
    axis_width = np.amax(points, axis=0) - np.amin(points, axis=0)
    axis = axis_width.argsort()[-1]

    AXES.pop(axis)

    long_axis_vs = np.sort(np.unique(points[:, axis]))

    dists = []
    for v in long_axis_vs:
        idices = np.where(points[:, axis] == v)
        tmp_points = points[idices][:, AXES]
        centroid = np.mean(tmp_points, axis=0)
        dist = np.squeeze(distance.cdist(tmp_points, centroid.reshape((1, -1))))

        dist = np.mean(dist[(np.quantile(dist, 0.1) < dist) & (dist < np.quantile(dist, 0.9))])

        dists.append(dist)

    dists = np.array(dists, dtype=float)

    # normalize
    dists -= np.amin(dists)
    dists /= np.amax(dists)

    starting_index, ending_index = guess_cortex_position(long_axis_vs, dists, factor, title=title, trial=0)

    idices = np.where((points[:, axis] >= long_axis_vs[starting_index])
                      & (points[:, axis] <= long_axis_vs[ending_index]))
    tmp_points = points[idices]

    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(tmp_points)

    if return_indices:
        return tmp_pcd, starting_index, ending_index

    return tmp_pcd


def prepare_bboxes(ref_pcd, sub_pcd, cortex=True):
    if cortex:
        ref_cortex, ref_AP = find_cortex_bbox(ref_pcd, factor=4)
        sub_cortex, sub_AP = find_cortex_bbox(sub_pcd, factor=10)
        ref_bbox = ref_cortex.get_axis_aligned_bounding_box()
        sub_bbox = sub_cortex.get_axis_aligned_bounding_box()
    else:
        ref_bbox = ref_pcd.get_axis_aligned_bounding_box()
        sub_bbox = sub_pcd.get_axis_aligned_bounding_box()

    return ref_bbox, sub_bbox


def estimate_scaling(ref_bbox, sub_bbox, ref_sorted_index=None, sub_sorted_index=None):
    ref_width = ref_bbox.get_extent()

    if isinstance(sub_bbox, o3d.geometry.AxisAlignedBoundingBox):
        sub_width = sub_bbox.get_extent()
    elif isinstance(sub_bbox, o3d.geometry.OrientedBoundingBox):
        sub_width = sub_bbox.extent
    else:
        raise RuntimeError("No such class type")

    trans_scaling = np.eye(4)
    for i in range(3):
        if not ref_sorted_index is None and not sub_sorted_index is None:
            idx = np.where(sub_sorted_index == i)
            trans_scaling[i, i] = ref_width[ref_sorted_index[idx]] / sub_width[i]
        else:
            trans_scaling[i, i] = ref_width[i] / sub_width[i]

    return trans_scaling


def estimate_translation(ref_bbox, sub_bbox):
    ref_center = ref_bbox.get_center()
    sub_center = sub_bbox.get_center()

    trans_translation = np.eye(4)
    for i in range(3):
        trans_translation[i, -1] = ref_center[i] - sub_center[i]

    return trans_translation


def estimate_rotation(ref_bbox, sub_bbox):
    ious = []
    rotations = []
    for (theta, phi, gamma) in itertools.product([0, 0.5, 1, 1.5], [0, 0.5, 1, 1.5], [0, 0.5, 1, 1.5]):
        tmp_bbox = sub_bbox.get_oriented_bounding_box()
        R = tmp_bbox.get_rotation_matrix_from_xyz((theta * np.pi, phi * np.pi, gamma * np.pi))
        tmp_bbox.rotate(R, center=tmp_bbox.get_center())

        rotations.append(R)
        ious.append(box3d_iou(ref_bbox, tmp_bbox.get_axis_aligned_bounding_box()))

    ious = np.array(ious)
    max_id = ious.argsort()[-1]

    trans_rotation = np.eye(4)
    trans_rotation[:3, :3] = rotations[max_id]

    return trans_rotation


def apply_transformation_return_bbox(pcd, transformation):
    pcd = pcd.transform(transformation)
    bbox = pcd.get_axis_aligned_bounding_box()

    return pcd, bbox


def iterative_estimate_transformation(ref_pcd, sub_pcd, max_iter=4, threshold=0.99, visualize=True):
    """return a list of transformations"""

    ref_sorted_index = (ref_pcd.get_max_bound() - ref_pcd.get_min_bound()).argsort()[::-1]
    sub_sorted_index = (sub_pcd.get_max_bound() - sub_pcd.get_min_bound()).argsort()[::-1]

    ref_cortex = find_cortex_bbox(ref_pcd, factor=4)
    sub_cortex = find_cortex_bbox(sub_pcd, factor=6)

    ref_bbox = ref_cortex.get_axis_aligned_bounding_box()
    sub_bbox = sub_cortex.get_axis_aligned_bounding_box()

    iou, iteration = 0, 0
    trans_list = []

    trans = estimate_scaling(ref_bbox, sub_bbox, ref_sorted_index, sub_sorted_index)
    sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
    trans_list.append(trans)

    while (iou <= threshold) & (iteration < max_iter):
        # step 0 calculate the IoU of two bounding boxes
        iou = box3d_iou(ref_bbox, sub_bbox)

        # step 1: translation to match center point of two bounding boxes
        trans = estimate_translation(ref_bbox, sub_bbox)
        sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
        trans_list.append(trans)
        # draw_only_bboxes(ref_bbox, sub_bbox)

        # step 2: rotation to match orientation
        trans = estimate_rotation(ref_bbox, sub_bbox)
        sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
        trans_list.append(trans)
        # draw_only_bboxes(ref_bbox, sub_bbox)

        # step 3: scaling to match size
        trans = estimate_scaling(ref_bbox, sub_bbox)
        sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
        trans_list.append(trans)
        # draw_only_bboxes(ref_bbox, sub_bbox)

        iteration += 1

    return trans_list


def iterative_estimate_transformation_by_cortex(ref_cortex, sub_cortex, max_iter=4, threshold=0.99, visualize=True):
    """return a list of transformations"""

    ref_sorted_index = (ref_cortex.get_max_bound() - ref_cortex.get_min_bound()).argsort()[::-1]
    sub_sorted_index = (sub_cortex.get_max_bound() - sub_cortex.get_min_bound()).argsort()[::-1]

    # ref_cortex = find_cortex_bbox(ref_pcd, factor=4)
    # sub_cortex = find_cortex_bbox(sub_pcd, factor=4)

    ref_bbox = ref_cortex.get_axis_aligned_bounding_box()
    sub_bbox = sub_cortex.get_axis_aligned_bounding_box()

    iou, iteration = 0, 0
    trans_list = []

    trans = estimate_scaling(ref_bbox, sub_bbox, ref_sorted_index, sub_sorted_index)
    sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
    # draw_only_bboxes(ref_bbox, sub_bbox)
    trans_list.append(trans)

    while (iou <= threshold) & (iteration < max_iter):
        # step 0 calculate the IoU of two bounding boxes
        iou = box3d_iou(ref_bbox, sub_bbox)

        # step 1: translation to match center point of two bounding boxes
        trans = estimate_translation(ref_bbox, sub_bbox)
        sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
        trans_list.append(trans)
        # draw_only_bboxes(ref_bbox, sub_bbox)

        # step 2: rotation to match orientation
        trans = estimate_rotation(ref_bbox, sub_bbox)
        sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
        trans_list.append(trans)
        # draw_only_bboxes(ref_bbox, sub_bbox)

        # step 3: scaling to match size
        trans = estimate_scaling(ref_bbox, sub_bbox)
        sub_cortex, sub_bbox = apply_transformation_return_bbox(sub_cortex, trans)
        trans_list.append(trans)
        # draw_only_bboxes(ref_bbox, sub_bbox)

        iteration += 1

    return trans_list


############################
###### Point Cloud  ########
############################
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def draw_point_cloud(pcd, transformation=None):
    pcd_temp = copy.deepcopy(pcd)
    pcd_temp.paint_uniform_color([1, 0.706, 0])
    if transformation is not None:
        pcd_temp.transform(transformation)
    o3d.visualization.draw_geometries([pcd_temp])


def draw_bboxes_with_source(source, bbox, bbox2):
    source_temp = copy.deepcopy(source)
    bbox_temp = copy.deepcopy(bbox)
    bbox2_temp = copy.deepcopy(bbox2)
    source_temp.paint_uniform_color([1, 0.706, 0])
    bbox_temp.color = [0, 0.651, 0.929]
    bbox2_temp.color = [1, 0.1, 0.123]
    o3d.visualization.draw_geometries([source_temp, bbox_temp, bbox2_temp])


def draw_only_bboxes(bbox, bbox2):
    bbox_temp = copy.deepcopy(bbox)
    bbox2_temp = copy.deepcopy(bbox2)
    bbox_temp.color = [1, 0, 0]
    bbox2_temp.color = [0, 1, 0]
    o3d.visualization.draw_geometries([bbox_temp, bbox2_temp])


def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        if isinstance(transformation, np.ndarray):
            source_temp.transform(transformation)
        elif isinstance(transformation, Transformation):  # probreg transformation type
            source_temp.points = transformation.transform(source_temp.points)
        else:
            source_temp.points = transformation[0].transform(source_temp.points)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


############################
########    Class   ########
############################
class GlobalContourRegister():
    """Global Contour Register"""

    def __init__(self, point_cloud_file, meta_file, downsampled_voxel_size=0.1, isref=False, ref_meta=None):
        super(GlobalContourRegister).__init__()
        if isref:
            self.meta_file = meta_file
        else:
            self.meta_file = ref_meta
        self.point_cloud_file = point_cloud_file
        self.downsampled_voxel_size = downsampled_voxel_size

        self.isref = isref

        self.scaling_matrix = self.get_voxel_scaling_matrix()

    def get_voxel_scaling_matrix(self):
        spacing = load_json(self.meta_file)['spacing']

        trans = np.eye(4)
        trans[0, 0] = spacing[0]
        trans[1, 1] = spacing[1]
        trans[2, 2] = spacing[2]

        return trans

    def load_point_cloud(self, eps=0.02, min_points=10, physical=True):
        print(":: Load point clouds and apply the initial pose.")

        pcd = o3d.io.read_point_cloud(self.point_cloud_file)

        # points = np.unique(np.array(pcd.points), axis=0)  # zyx
        # points = np.stack((points[:, 2], points[:, 1], points[:, 0]), axis=-1)  # xyz
        # pcd.points = o3d.utility.Vector3dVector(points)

        # apply scaling to physical size according to the meta_data
        if physical:  # convert to physical point cloud space
            pcd = pcd.transform(self.scaling_matrix)
            # pass

        # TODO: remove isolated points, this can be done by refining the segmentation
        return pcd

    def compute_fpfh_features(self, pcd):
        print(":: Downsample with a voxel size %.3f." % self.downsampled_voxel_size)
        pcd_down = pcd.voxel_down_sample(self.downsampled_voxel_size)

        radius_normal = self.downsampled_voxel_size * 10
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=1000))

        radius_feature = self.downsampled_voxel_size * 20
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=2000))

        # draw_point_cloud(pcd_down)
        return pcd_down, pcd_fpfh


class NonRigidRegister():
    """
    Algorithm type('cpd', 'svr', 'gmmreg', 'bcpd')
    """

    def __init__(self, algo_type_name: str = "cpd"):
        super(NonRigidRegister).__init__()
        self.algo_type_name = algo_type_name

    def __call__(self, source, target, **kwargs):
        cv = lambda x: np.asarray(x.points if isinstance(x, o3d.geometry.PointCloud) else x)
        if self.algo_type_name == "cpd":
            reg = cpd.NonRigidCPD(cv(source))
        elif self.algo_type_name == "svr":
            reg = l2dist_regs.TPSSVR(cv(source))
        elif self.algo_type_name == "gmmreg":
            reg = l2dist_regs.TPSGMMReg(cv(source))
        elif self.algo_type_name == "bcpd":
            reg = bcpd.CombinedBCPD(cv(source))
        else:
            raise ValueError("Unknown algorithm type %s" % self.algo_type_name)

        res = reg.registration(cv(target))

        return res
