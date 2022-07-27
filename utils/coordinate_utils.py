import numpy as np


def scale_coords(coords, scale):
    """
    :param coords: numpy array
                    N x 3
    :param scale:
                    1 x 3
    :return:
            numpy array N x 3
    """

    coords = coords.astype(float)
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == coords.shape[1]
        for i in range(len(scale)):
            coords[:, i] *= scale[i]
    else:
        coords *= scale
    return coords


def contours_to_points(contours, axis=0):
    """
    :param contours:
    :param axis:
    :return:
        points:
            N x 3
    """
    points = []
    for i, contour_list in enumerate(contours):
        if len(contour_list) > 0:
            for contour in contour_list:
                rep_i = np.asarray(i).repeat(len(contour))
                c0 = np.array(contour[:, 0], dtype=np.int64)
                c1 = np.array(contour[:, 1], dtype=np.int64)

                if axis == 0:
                    tmp = np.stack([rep_i, c0, c1], axis=1)
                elif axis == 1:
                    tmp = np.stack([c0, rep_i, c1], axis=1)
                else:
                    tmp = np.stack([c0, c1, rep_i], axis=1)

                points.append(tmp)

    points = np.concatenate(points, axis=0)

    return points
