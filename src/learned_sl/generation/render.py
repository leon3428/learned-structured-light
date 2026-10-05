"""Render LTM, depth and occlusion for every mesh with the modified Mitsuba 3.

Requires the Mitsuba fork in ``external/mitsuba`` built with the ``cuda_mono`` variant.

sl-generate -i data/abc_meshes -o data/abc_diffuse --object-material diffuse
sl-split --dataset data/abc_diffuse

Work is split into chunks recorded in ``chunks.json``; each invocation renders the next
free chunk, so several processes or machines can share one output folder. Each sample is
written to ``<output>/<uuid>/{ltm,depth,occlusion}.h5``. The published dataset was then
packed into Parquet shards with the scripts in its Hugging Face repository.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import mitsuba as mi
import numpy as np
from PIL import Image
from tqdm import tqdm

from learned_sl.data import RESOLUTION
from learned_sl.generation.scene import create_scene

SAMPLE_COUNT = 32
CHUNK_SIZE = 100_000
LTM_RR_DEPTH = 5
LTM_CLAMP = 1e-4  # path-sample contributions <= this are dropped
OCCLUSION_MAX_DEPTH = 2  # direct illumination only
OCCLUSION_THRESHOLD = 1e-3
PREVIEW_MAX_DEPTH = 8
CHUNKS_FILENAME = "chunks.json"
META_FILENAME = "meta.json"


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_chunks(output: Path, model_ids: list[str], chunk_size: int) -> list[dict]:
    path = output / CHUNKS_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    chunks = [
        {"status": "free", "model_ids": model_ids[i : i + chunk_size]}
        for i in range(0, len(model_ids), chunk_size)
    ]
    _write_json(path, chunks)
    return chunks


def _save_h5(path: Path, name: str, data) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=data, compression="gzip", compression_opts=1)


def render_sample(generator, scene, sample_dir: Path) -> None:
    generator.render(str(sample_dir / "ltm.h5"), scene)
    _save_h5(sample_dir / "depth.h5", "depth", mi.render(scene))
    direct = mi.load_dict({"type": "path", "max_depth": OCCLUSION_MAX_DEPTH})
    occlusion = mi.render(scene, integrator=direct).numpy() > OCCLUSION_THRESHOLD
    _save_h5(sample_dir / "occlusion.h5", "occlusion", occlusion)


def generate_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-folder", type=Path, required=True, help="output of sl-abc-prepare"
    )
    parser.add_argument("-o", "--output-folder", type=Path, required=True)
    parser.add_argument("--object-material", choices=("diffuse", "metallic"), default="diffuse")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    mi.set_variant("cuda_mono")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.input_folder / META_FILENAME).read_text())
    chunks = _load_chunks(args.output_folder, list(meta), args.chunk_size)

    index = next((i for i, c in enumerate(chunks) if c["status"] == "free"), None)
    if index is None:
        print("All chunks are processed")
        return
    print(f"Processing chunk {index + 1}/{len(chunks)}")
    chunks[index]["status"] = "processing"
    _write_json(args.output_folder / CHUNKS_FILENAME, chunks)

    generator = mi.DatasetGenerator(
        LTM_RR_DEPTH, False, args.sample_count, RESOLUTION, RESOLUTION, 1, LTM_CLAMP
    )
    for model_id in tqdm(chunks[index]["model_ids"]):
        sample_dir = args.output_folder / str(uuid4())
        sample_dir.mkdir(parents=True)
        scene = create_scene(
            args.input_folder / model_id,
            RESOLUTION,
            args.sample_count,
            meta[model_id],
            object_material=args.object_material,
        )
        render_sample(generator, scene, sample_dir)

    chunks[index]["status"] = "done"
    _write_json(args.output_folder / CHUNKS_FILENAME, chunks)


def preview_main() -> None:
    parser = argparse.ArgumentParser(description="Path-trace a preview image of one scene.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="output of sl-abc-prepare")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("preview.png"))
    parser.add_argument("--object-material", choices=("diffuse", "metallic"), default="diffuse")
    args = parser.parse_args()

    mi.set_variant("cuda_mono")
    random.seed(args.seed)
    meta = json.loads((args.dataset_path / META_FILENAME).read_text())
    model_id = args.model_id or next(iter(meta))
    scene = create_scene(
        args.dataset_path / model_id,
        RESOLUTION,
        args.sample_count,
        meta[model_id],
        object_material=args.object_material,
    )
    integrator = mi.load_dict({"type": "path", "max_depth": PREVIEW_MAX_DEPTH})
    image = mi.render(scene, integrator=integrator).numpy() ** (1.0 / 2.2)
    image = np.minimum(image * 255, 255).astype(np.uint8).squeeze()
    Image.fromarray(image).save(args.output)
    print(f"{model_id} -> {args.output}")


def split_main() -> None:
    parser = argparse.ArgumentParser(description="Write split.json with a train/test split.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.9, help="train fraction")
    args = parser.parse_args()

    samples = sorted(p.name for p in args.dataset.iterdir() if (p / "ltm.h5").exists())
    cut = int(args.ratio * len(samples))
    print(f"Total {len(samples)}, train {cut}, test {len(samples) - cut}")
    _write_json(args.dataset / "split.json", {"train": samples[:cut], "test": samples[cut:]})
