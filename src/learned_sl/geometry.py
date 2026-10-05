"""Simulated camera-projector geometry (paper Fig. 2) and triangulation.

The same constants build the Mitsuba scenes and the classical baselines' calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

FOV_DEG = 10.0  # horizontal field of view of both camera and projector
CAMERA_POSITION = (0.0, 0.0, -1.0)  # 1 m above the table, looking along +z
PROJECTOR_POSITION = (-0.1, 0.0, -1.0)  # 0.1 m baseline
PROJECTOR_YAW_DEG = 5.7  # rotation about the y axis, toward the camera's view


def intrinsic_matrix(image_resolution: tuple[int, int], fov_deg: float = FOV_DEG):
    """Ideal pinhole intrinsics matching the synthetic camera/projector."""
    width, height = image_resolution
    focal = width / (2 * np.tan(np.radians(fov_deg) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


def projector_pose() -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation of the projector relative to the camera."""
    angle = np.radians(PROJECTOR_YAW_DEG)
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    translation = np.array([PROJECTOR_POSITION[0] - CAMERA_POSITION[0], 0, 0], dtype=np.float64)
    return rotation, translation


def triangulate(K1, K2, R, t, x1, y1, x2):
    """Triangulate camera pixels (x1, y1) with projector columns x2.

    The projector row is inferred from the epipolar line of each camera pixel.
    Returns the 3D points and the mask of correspondences that could be triangulated.
    """
    n_points = len(x1)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])

    points1 = np.column_stack([x1, y1]).astype(np.float32)
    points2 = np.full((n_points, 2), np.nan, dtype=np.float32)
    points2[:, 0] = x2

    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K2).T @ (t_cross @ R) @ np.linalg.inv(K1)

    valid = np.isfinite(x2)
    lines = (F @ np.column_stack([x1, y1, np.ones(n_points)]).T).T
    valid &= np.abs(lines[:, 1]) > 1e-10
    points2[valid, 1] = -(lines[valid, 0] * x2[valid] + lines[valid, 2]) / lines[valid, 1]

    if not valid.any():
        return np.zeros((0, 3)), valid

    points_4d = cv2.triangulatePoints(P1, P2, points1[valid].T, points2[valid].T)
    return (points_4d[:3] / points_4d[3]).T, valid


def point_cloud_to_depth_map(points_3d, pixels, image_shape: tuple[int, int]):
    """Rasterize points into a z-depth map, keeping the nearest point per pixel."""
    height, width = image_shape
    depth = np.full((height, width), np.nan, dtype=np.float32)
    hits = np.zeros((height, width), dtype=int)

    x = np.round(pixels[:, 0]).astype(int)
    y = np.round(pixels[:, 1]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    for xi, yi, z in zip(x[inside], y[inside], points_3d[inside, 2]):
        if np.isnan(depth[yi, xi]) or z < depth[yi, xi]:
            depth[yi, xi] = z
        hits[yi, xi] += 1
    return depth, hits
