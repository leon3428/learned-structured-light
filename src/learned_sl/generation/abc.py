"""Download and prepare ABC meshes (https://deep-geometry.github.io/abc-dataset/).

sl-abc-urls data/abc_raw              # writes data/abc_raw/abc_obj_urls.txt
# download the listed .7z chunks into data/abc_raw (e.g. with wget -i)
sl-abc-prepare -i data/abc_raw -o data/abc_meshes

Each mesh is recentred on its bounding-box centre and written to
``<output>/<uuid>/mesh.ply``; ``<output>/meta.json`` records its original name and
bounding-box size. The published dataset rendered the first 100,000 of 105,903 meshes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

import py7zr
import requests
import trimesh
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://archive.nyu.edu"
INDEX_URL = f"{BASE_URL}/handle/2451/43778?rpp=100&value=1&data-order=asc"


def _links(session: requests.Session, url: str):
    response = session.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser").select("a[href]")


def list_urls_main() -> None:
    parser = argparse.ArgumentParser(description="List the download URLs of the ABC OBJ chunks.")
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()

    session = requests.Session()
    chunk_pages = sorted(
        {
            BASE_URL + a["href"]
            for a in _links(session, INDEX_URL)
            if a.text.strip().startswith("ABC Dataset Chunk")
        }
    )
    print(f"Found {len(chunk_pages)} chunk pages")

    urls = []
    for i, page in enumerate(chunk_pages, 1):
        print(f"[{i}/{len(chunk_pages)}] {page}")
        urls += [
            BASE_URL + a["href"] for a in _links(session, page) if "obj" in a.text.strip().lower()
        ][:1]
        time.sleep(0.5)  # be nice to the server

    args.output_folder.mkdir(parents=True, exist_ok=True)
    path = args.output_folder / "abc_obj_urls.txt"
    path.write_text("".join(url + "\n" for url in urls))
    print(f"Saved {len(urls)} URLs to {path}")


def prepare_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-folder", type=Path, required=True, help="folder with ABC .7z chunks"
    )
    parser.add_argument("-o", "--output-folder", type=Path, required=True)
    parser.add_argument(
        "--chunks", nargs="*", default=None, help="chunk names to process (default: all)"
    )
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    temp = args.input_folder / "temp"

    for archive_path in sorted(args.input_folder.glob("*.7z")):
        chunk = archive_path.name.split(".")[0]
        if args.chunks is not None and chunk not in args.chunks:
            continue
        print(f"Chunk {chunk}")
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                archive.extractall(path=temp / chunk)
        except py7zr.Bad7zFile as e:
            print(f"Skipping {archive_path}: {e}")
            continue

        for obj_path in tqdm(sorted((temp / chunk).glob("*/*.obj"))):
            mesh = trimesh.load(obj_path)
            if mesh.is_empty:
                continue
            mesh.apply_translation(-mesh.bounding_box.centroid)

            model_id = str(uuid4())
            (args.output_folder / model_id).mkdir()
            mesh.export(args.output_folder / model_id / "mesh.ply")
            x, y, z = (float(v) for v in mesh.extents)
            meta[model_id] = {"original_name": obj_path.stem, "x": x, "y": y, "z": z}

        shutil.rmtree(temp)
        meta_path.write_text(json.dumps(meta, indent=4))
    print(f"{len(meta)} meshes in {meta_path}")
