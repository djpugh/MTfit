"""
spherical_projection
********************

Spherical projection functions:
    * Equal Area
    * Equal Angle
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import numpy as np


def equal_area(x, y, z, projection_axis=False, lower=True, full_sphere=False, back_project=False):
    """
    Equal area projection of cartesian points on a sphere to cartesian points on a plane.

    Args:
        x: array-like - array of spherical cartesian coordinates to transform
        y: array-like - array of spherical cartesian coordinates to transform
        z: array-like - array of spherical cartesian coordinates to transform


    Keyword Args:
        projection_axis: array-like - vector of the projection axis
        lower: boolean - boolean flag for lower or upper hemisphere projection
        full_sphere: boolean - boolean flag for full sphere projection
        back_project: boolean - boolean flag for back projecting the results

    Returns:
        (np.array, np.array) - arrays of corresponding projected X and Y coordinates
    """

    def equal_area_correction(x, y, z, lower=True):
        if lower:
            return np.sqrt(2/(1+z))
        return np.sqrt(2/(1-z))
    return __project(x, y, z, equal_area_correction, projection_axis, lower, full_sphere, back_project)


def equal_angle(x, y, z, projection_axis=False, lower=True, full_sphere=False, back_project=False):
    """
    Equal angle (stereographic) projection of cartesian points on a sphere to cartesian points on a plane.

    Args:
        x: array-like - array of spherical cartesian coordinates to transform
        y: array-like - array of spherical cartesian coordinates to transform
        z: array-like - array of spherical cartesian coordinates to transform


    Keyword Args:
        projection_axis: array-like - vector of the projection axis
        lower: boolean - boolean flag for lower or upper hemisphere projection
        full_sphere: boolean - boolean flag for full sphere projection
        back_project: boolean - boolean flag for back projecting the results

    Returns:
        (np.array, np.array) - arrays of corresponding projected X and Y coordinates
    """

    def equal_angle_correction(x, y, z, lower=True):
        if lower:
            return 1/(1+z)
        return 1/(1-z)
    return __project(x, y, z, equal_angle_correction, projection_axis, lower, full_sphere, back_project)


def __project(x, y, z, correction_function, projection_axis=False, lower=True, full_sphere=False, back_project=False):
    """
    Helper function for projection of cartesian points on a sphere to cartesian points on a plane.

    Args:
        x: array-like - array of spherical cartesian coordinates to transform
        y: array-like - array of spherical cartesian coordinates to transform
        z: array-like - array of spherical cartesian coordinates to transform
        correction_function: scaling correction based on the projection

    Keyword Args:
        projection_axis: array-like - vector of the projection axis
        lower: boolean - boolean flag for lower or upper hemisphere projection
        full_sphere: boolean - boolean flag for full sphere projection
        back_project: boolean - boolean flag for back projecting the results

    Returns:
        (np.array, np.array) - arrays of corresponding projected X and Y coordinates
    """

    if not isinstance(x, np.ndarray):
        x = np.array([x])
    if not isinstance(y, np.ndarray):
        y = np.array([y])
    if not isinstance(z, np.ndarray):
        z = np.array([z])
    if not len(x.shape):
        x = np.array([x])
    if not len(y.shape):
        y = np.array([y])
    if not len(z.shape):
        z = np.array([z])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    if isinstance(projection_axis, (list, np.ndarray)):
        projection_axis = np.array(projection_axis).flatten()
        rAxis = np.cross(projection_axis, np.array([0, 0, 1]))
        theta = - np.arcsin(np.sqrt(np.sum(rAxis ^ 2)) /
                            (np.sqrt(np.sum(projection_axis ^ 2))))
        rAxis = rAxis/np.sqrt(np.sum(rAxis ^ 2))
        R = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])*np.cos(theta) + \
            np.sin(theta)*np.matrix([[0, -rAxis[2], rAxis[1]],
                                     [rAxis[2], 0, -rAxis[0]],
                                     [-rAxis[1], rAxis[0], 0]]) + \
            (1-np.cos(theta))*(np.matrix(rAxis).T*np.matrix(rAxis))
        V = R*np.matrix([[x], [y], [z]])
        x = V[1, :]
        y = V[2, :]
        z = V[3, :]
    X = x*correction_function(x, y, z, lower)
    Y = y*correction_function(x, y, z, lower)
    if not full_sphere:
        if lower:
            X[z < 0] = np.nan
            Y[z < 0] = np.nan
        else:
            X[z > 0] = np.nan
            Y[z > 0] = np.nan
    if back_project:
        X_b = -x*correction_function(-x, -y, -z, lower)
        Y_b = -y*correction_function(-x, -y, -z, lower)
        X[np.isnan(X)] = X_b[np.isnan(Y)]
        Y[np.isnan(Y)] = Y_b[np.isnan(Y)]
    X[np.isinf(X)] = np.nan
    Y[np.isinf(Y)] = np.nan
    return X, Y
