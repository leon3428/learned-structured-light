import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from learned_sl import data

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("depth", pa.list_(pa.list_(pa.float32()))),
        ("occlusion", pa.list_(pa.list_(pa.bool_()))),
        ("ltm_rows", pa.list_(pa.uint32())),
        ("ltm_cols", pa.list_(pa.uint32())),
        ("ltm_values", pa.list_(pa.uint16())),
    ]
)


def make_row(sample_id: str, rng: np.random.Generator) -> dict:
    """A sample whose LTM maps a few camera pixels to random projector pixels."""
    rows = np.sort(rng.choice(data.PIXELS, size=5, replace=False)).astype(np.uint32)
    cols = rng.integers(0, data.PIXELS, size=5).astype(np.uint32)
    values = rng.integers(1, 65535, size=5).astype(np.uint16)
    depth = rng.uniform(0.8, 1.0, size=data.RESOLUTION).astype(np.float32)
    occlusion = rng.random(data.RESOLUTION) > 0.1
    return {
        "sample_id": sample_id,
        "depth": depth.tolist(),
        "occlusion": occlusion.tolist(),
        "ltm_rows": rows,
        "ltm_cols": cols,
        "ltm_values": values,
    }


def write_shards(root, subset, split, ids, per_shard, per_group, seed=0):
    rng = np.random.default_rng(seed)
    shards = [ids[i : i + per_shard] for i in range(0, len(ids), per_shard)]
    (root / subset).mkdir(parents=True, exist_ok=True)
    for k, shard in enumerate(shards):
        path = root / subset / f"{split}-{k:05d}-of-{len(shards):05d}.parquet"
        with pq.ParquetWriter(path, SCHEMA) as writer:
            for start in range(0, len(shard), per_group):
                rows = [make_row(i, rng) for i in shard[start : start + per_group]]
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """24 train samples (6 held out for validation) and 10 test samples."""
    train_ids = [f"train-{i:02d}" for i in range(24)]
    test_ids = [f"test-{i:02d}" for i in range(10)]
    write_shards(tmp_path, "diffuse", "train", train_ids, per_shard=8, per_group=4)
    write_shards(tmp_path, "diffuse", "test", test_ids, per_shard=10, per_group=3, seed=1)
    val_ids = train_ids[1::4]
    monkeypatch.setattr(data, "validation_ids", lambda subset: list(val_ids))
    return tmp_path, train_ids, val_ids, test_ids
