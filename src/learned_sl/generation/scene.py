"""Mitsuba scene for one dataset sample: camera, projector, table and one ABC mesh."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import mitsuba as mi

from learned_sl.geometry import CAMERA_POSITION, FOV_DEG, PROJECTOR_POSITION, PROJECTOR_YAW_DEG

ObjectMaterial = Literal["diffuse", "metallic"]

BACKGROUND_MESH = Path(__file__).with_name("background.ply")
TABLE_REFLECTANCE = 0.3
OBJECT_DIFFUSE_REFLECTANCE = 0.6
METALLIC_MATERIAL = "Al"
METALLIC_ROUGHNESS = 0.15
PROJECTOR_SCALE = 0.3
FAR_CLIP = 2.5


def object_bsdf(material: ObjectMaterial) -> dict[str, Any]:
    if material == "metallic":
        bsdf: dict[str, Any] = {
            "type": "roughconductor",
            "material": METALLIC_MATERIAL,
            "distribution": "ggx",
            "alpha": METALLIC_ROUGHNESS,
        }
    else:
        bsdf = {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": OBJECT_DIFFUSE_REFLECTANCE},
        }
    return {"type": "twosided", "material": bsdf}


def table_bsdf() -> dict[str, Any]:
    return {
        "type": "twosided",
        "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": TABLE_REFLECTANCE}},
    }


def create_scene(
    mesh_dir: str | Path,
    sensor_size: tuple[int, int],
    sample_count: int,
    extents: dict,
    object_material: ObjectMaterial = "diffuse",
):
    """Scene with the mesh randomly scaled, shifted and rotated on the table.

    ``extents`` holds the mesh bounding-box size as ``{"x": ..., "y": ..., "z": ...}``.
    """
    camera_transform = mi.ScalarTransform4f().translate(list(CAMERA_POSITION))
    projector_transform = (
        mi.ScalarTransform4f()
        .translate(list(PROJECTOR_POSITION))
        .rotate(axis=[0, 1, 0], angle=PROJECTOR_YAW_DEG)
    )

    max_dim = max(extents["x"], extents["y"], extents["z"])
    scale = random.uniform(0.12, 0.17) / max_dim
    offset_x = random.uniform(-0.05, 0.05)
    offset_y = random.uniform(-0.05, 0.05)
    offset_z = -extents["z"] * scale / 2
    rotation = random.uniform(0, 360)
    model_transform = (
        mi.ScalarTransform4f()
        .translate([offset_x, offset_y, offset_z])
        .rotate(axis=[0, 1, 0], angle=180)
        .rotate(axis=[0, 0, 1], angle=rotation)
        .scale(scale)
    )

    return mi.load_dict(
        {
            "type": "scene",
            "myintegrator": {"type": "zdepth"},
            "mysensor": {
                "type": "perspective",
                "near_clip": 0.001,
                "far_clip": FAR_CLIP,
                "fov_axis": "x",
                "fov": FOV_DEG,
                "principal_point_offset_x": 0.0,
                "principal_point_offset_y": 0.0,
                "to_world": camera_transform,
                "mysampler": {"type": "independent", "sample_count": sample_count},
                "myfilm": {
                    "type": "hdrfilm",
                    "pixel_format": "luminance",
                    "width": sensor_size[0],
                    "height": sensor_size[1],
                },
            },
            "myemitter": {
                "type": "projector",
                "irradiance": {"type": "rgb", "value": 1.0},
                "scale": PROJECTOR_SCALE,
                "fov_axis": "x",
                "fov": FOV_DEG,
                "to_world": projector_transform,
            },
            "table": {"type": "ply", "filename": str(BACKGROUND_MESH), "material": table_bsdf()},
            "object": {
                "type": "ply",
                "filename": str(Path(mesh_dir) / "mesh.ply"),
                "material": object_bsdf(object_material),
                "to_world": model_transform,
            },
        }
    )
