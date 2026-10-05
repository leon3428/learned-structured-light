from collections import Counter

import numpy as np
import pytest
import torch

from learned_sl import data
from learned_sl.data import LtmBatches, LtmSplit, collate, read_row_group


def test_splits_partition_samples(tiny_dataset):
    root, train_ids, val_ids, test_ids = tiny_dataset

    train = LtmSplit(root, "diffuse", "train")
    val = LtmSplit(root, "diffuse", "val")
    test = LtmSplit(root, "diffuse", "test")

    assert set(val.sample_ids) == set(val_ids)
    assert set(train.sample_ids) == set(train_ids) - set(val_ids)
    assert test.sample_ids == test_ids
    assert [s.sample_id for g in val.groups for s in read_row_group(g)] == val.sample_ids


def test_missing_shards_are_an_error_unless_allowed(tiny_dataset):
    root, *_ = tiny_dataset
    next((root / "diffuse").glob("train-00001-*.parquet")).unlink()

    with pytest.raises(FileNotFoundError, match="2 of 3"):
        LtmSplit(root, "diffuse", "train")
    assert len(LtmSplit(root, "diffuse", "train", allow_partial=True)) > 0


def test_collate_stacks_ltms_by_camera_rows(tiny_dataset):
    root, *_ = tiny_dataset
    samples = read_row_group(LtmSplit(root, "diffuse", "test").groups[0])[:2]

    ltms, depths, occlusions = collate(samples)

    assert ltms.shape == (2 * data.PIXELS, data.PIXELS)
    assert ltms.is_coalesced()
    dense = ltms.to_dense()
    for i, s in enumerate(samples):
        expected = s.values.astype(np.float32) / 65535
        actual = dense[i * data.PIXELS + s.rows.astype(np.int64), s.cols.astype(np.int64)]
        assert torch.equal(actual, torch.from_numpy(expected))
    assert depths.shape == occlusions.shape == (2, 1, 256, 256)
    assert occlusions.dtype == torch.bool
    assert np.array_equal(depths[1, 0].numpy(), samples[1].depth)


def _ids(batches: LtmBatches, workers: int) -> list[list[str]]:
    """Sample ids yielded by each simulated worker, recovered from depth maps."""
    by_depth = {
        s.depth.tobytes(): s.sample_id for g in batches.split.groups for s in read_row_group(g)
    }
    result = []
    for worker in range(workers):
        info = type("Info", (), {"id": worker, "num_workers": workers})()
        original = data.get_worker_info
        data.get_worker_info = lambda info=info: info
        try:
            ids = [
                by_depth[depth[0].numpy().tobytes()] for _, depths, _ in batches for depth in depths
            ]
        finally:
            data.get_worker_info = original
        result.append(ids)
    return result


@pytest.mark.parametrize("world_size,workers", [(1, 1), (2, 1), (2, 3)])
def test_evaluation_yields_every_sample_once(tiny_dataset, world_size, workers):
    root, *_ = tiny_dataset
    split = LtmSplit(root, "diffuse", "test")

    seen = Counter()
    for rank in range(world_size):
        batches = LtmBatches(
            split,
            4,
            shuffle=False,
            drop_last=False,
            rank=rank,
            world_size=world_size,
            num_workers=workers,
        )
        seen.update(i for worker in _ids(batches, workers) for i in worker)

    assert seen == Counter(split.sample_ids)


def test_training_ranks_get_equal_full_batches_and_reshuffle(tiny_dataset):
    root, *_ = tiny_dataset
    split = LtmSplit(root, "diffuse", "train")

    def epoch_ids(epoch):
        per_rank = []
        for rank in range(2):
            batches = LtmBatches(
                split, 2, shuffle=True, drop_last=True, rank=rank, world_size=2, num_workers=2
            )
            batches.set_epoch(epoch)
            ids = [i for worker in _ids(batches, 2) for i in worker]
            assert len(ids) == 2 * len(batches)
            per_rank.append(ids)
        return per_rank

    first = epoch_ids(0)
    assert len(first[0]) == len(first[1]) > 0
    assert not set(first[0]) & set(first[1])
    assert epoch_ids(0) == first
    assert epoch_ids(1) != first


def test_validation_id_files_match_paper_split():
    for subset in data.SUBSETS:
        ids = data.validation_ids(subset)
        assert len(ids) == len(set(ids)) == 9000
