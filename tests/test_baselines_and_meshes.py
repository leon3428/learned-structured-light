import numpy as np
import pytest
from PIL import Image
from scipy.sparse import identity

from learned_sl import baselines
from learned_sl.figures.meshes import (
    DepthRenderConfig,
    depth_to_mesh,
    face_colors,
    render_depth_mesh,
)
from learned_sl.geometry import intrinsic_matrix, projector_pose, triangulate


def identity_ltm(resolution):
    width, height = resolution
    return identity(width * height, format="coo", dtype=np.float64)


def test_phase_shift_decoding_recovers_projector_columns():
    resolution = (64, 4)
    expected = np.tile(np.arange(64, dtype=np.float64), (4, 1))

    for steps in (3, 5):
        decoded = baselines.decode_phase_shift(
            baselines.phase_shift_patterns(resolution, steps), 64
        )
        wrapped_error = np.mod(decoded - expected + 32, 64) - 32
        assert np.abs(wrapped_error).max() < 0.25


@pytest.mark.parametrize("method", list(baselines.METHODS))
def test_correspondence_without_scene_effects_is_the_identity(method):
    resolution = (256, 256)
    expected = np.tile(np.arange(256, dtype=np.float64), (256, 1))

    decoded = baselines.correspondence(method, identity_ltm(resolution), resolution)

    assert np.nanmean(np.abs(decoded - expected)) < 1.0


def test_triangulation_inverts_projection():
    K = intrinsic_matrix((256, 256))
    R, t = projector_pose()
    point = np.array([0.01, -0.02, 0.95])
    camera_pixel = K @ point
    projector_pixel = K @ (R @ point + t)
    x1, y1 = camera_pixel[:2] / camera_pixel[2]
    x2 = projector_pixel[0] / projector_pixel[2]

    points, valid = triangulate(K, K, R, t, np.array([x1]), np.array([y1]), np.array([x2]))

    assert valid.all()
    assert np.allclose(points[0], point, atol=1e-5)


def test_depth_to_mesh_builds_faces_only_between_valid_pixels():
    depth = np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32)
    mesh = depth_to_mesh(depth, np.ones_like(depth, dtype=bool))
    assert mesh.vertices.shape == (4, 3) and mesh.faces.shape == (2, 3)

    depth[0, 1] = np.nan
    mesh = depth_to_mesh(depth, np.ones_like(depth, dtype=bool))
    assert mesh.vertices.shape == (3, 3) and mesh.faces.shape == (0, 3)


def test_depth_to_mesh_downsamples_and_skips_depth_jumps():
    mesh = depth_to_mesh(np.ones((8, 8)), None, DepthRenderConfig(max_mesh_size=4))
    assert mesh.vertices.shape == (16, 3) and mesh.faces.shape == (18, 3)

    jump = np.array([[1.0, 1.0], [1.0, 2.0]])
    mesh = depth_to_mesh(jump, None, DepthRenderConfig(max_face_depth_delta=0.25))
    assert mesh.faces.shape == (0, 3)


def test_metric_mode_back_projects_vertices():
    config = DepthRenderConfig(height_mode="metric", height_scale=1.0, fov_deg=90.0)

    mesh = depth_to_mesh(np.ones((2, 2)), None, config)

    expected = [[-1.0, 1.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    assert np.allclose(mesh.vertices, expected)


def test_specular_highlight_brightens_faces():
    mesh = depth_to_mesh(np.ones((2, 2)), None)
    dull = DepthRenderConfig(material_color=(0.2, 0.2, 0.2), specular_strength=0.0)
    shiny = DepthRenderConfig(material_color=(0.2, 0.2, 0.2), specular_strength=0.8, shininess=1.0)

    assert face_colors(mesh, shiny).max() > face_colors(mesh, dull).max()


def test_render_writes_png(tmp_path):
    depth = 0.93 + 0.01 * np.random.default_rng(0).random((16, 16))
    path = tmp_path / "mesh.png"

    render_depth_mesh(depth, np.ones_like(depth, dtype=bool), path, DepthRenderConfig(size=200))

    image = np.asarray(Image.open(path))
    assert image.shape[:2] == (200, 200)
    assert len(np.unique(image.reshape(-1, image.shape[-1]), axis=0)) > 1
