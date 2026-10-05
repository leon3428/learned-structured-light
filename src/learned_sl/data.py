"""Reader for the ABC Light Transport 256 dataset.

The dataset is published at https://huggingface.co/datasets/leon3428/abc-light-transport-256
as Parquet shards ``<subset>/{train,test}-XXXXX-of-YYYYY.parquet`` with 512 samples per
shard, stored in row groups of 64 samples. A row group is the smallest unit Parquet can
decode, so training shuffles the order of row groups and then shuffles samples inside a
small buffer of row groups.

The paper holds out 10% of the training split for validation (81,000 / 9,000). The
held-out sample ids are listed in ``splits/<subset>_val.txt``. They are the samples that
``torch.utils.data.random_split(train, [81000, 9000], generator=manual_seed(42))``
selected in the original training code.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from importlib import resources
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ID = "leon3428/abc-light-transport-256"
SUBSETS = ("diffuse", "metalic")
SPLITS = ("train", "val", "test")
DEFAULT_ROOT = "data/abc-light-transport-256"

# Camera and projector are both 256 x 256 in every sample.
RESOLUTION = (256, 256)
PIXELS = RESOLUTION[0] * RESOLUTION[1]
# LTM values are stored as uint16 fixed point.
LTM_SCALE = (1 << 16) - 1

# LTM indices are sorted and unique, so the sparse tensor invariant checks are skipped.
torch.sparse.check_sparse_tensor_invariants.disable()

_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.parquet$")


@dataclass
class Sample:
    sample_id: str
    rows: np.ndarray  # uint32, camera pixel index y * 256 + x
    cols: np.ndarray  # uint32, projector pixel index v * 256 + u
    values: np.ndarray  # uint16, transport coefficient * 65535
    depth: np.ndarray  # float32 (256, 256), camera z-depth in meters
    occlusion: np.ndarray  # bool (256, 256), True where lit by the projector


@dataclass(frozen=True)
class RowGroup:
    path: str
    index: int
    keep: tuple[bool, ...]  # which rows of the group belong to the split

    @property
    def num_samples(self) -> int:
        return sum(self.keep)


def validation_ids(subset: str) -> list[str]:
    text = resources.files("learned_sl").joinpath(f"splits/{subset}_val.txt")
    return text.read_text().split()


def _shard_files(root: Path, subset: str, split: str, allow_partial: bool) -> list[Path]:
    files = sorted((root / subset).glob(f"{split}-*-of-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {split} shards for {subset!r} under {root / subset}. "
            f"Download them with: sl-download --subset {subset} --root {root}"
        )
    expected = int(_SHARD_RE.search(files[0].name).group(2))
    if len(files) != expected and not allow_partial:
        raise FileNotFoundError(
            f"Found {len(files)} of {expected} {split} shards for {subset!r} under "
            f"{root / subset}. Finish the download or pass allow_partial=True."
        )
    return files


def _list_column(table, name: str) -> list[np.ndarray]:
    array = table.column(name).combine_chunks()
    offsets = array.offsets.to_numpy()
    values = array.values.to_numpy()
    return [values[a:b] for a, b in pairwise(offsets)]


def _image_column(table, name: str) -> np.ndarray:
    array = table.column(name).combine_chunks().flatten().flatten()
    width, height = RESOLUTION
    return array.to_numpy(zero_copy_only=False).reshape(-1, height, width)


def read_row_group(group: RowGroup) -> list[Sample]:
    table = pq.ParquetFile(group.path).read_row_group(group.index)
    ids = table.column("sample_id").to_pylist()
    rows = _list_column(table, "ltm_rows")
    cols = _list_column(table, "ltm_cols")
    values = _list_column(table, "ltm_values")
    depth = _image_column(table, "depth")
    occlusion = _image_column(table, "occlusion")
    return [
        Sample(ids[i], rows[i], cols[i], values[i], depth[i], occlusion[i])
        for i in range(len(ids))
        if group.keep[i]
    ]


class LtmSplit:
    """The row groups that make up one split of one subset."""

    def __init__(
        self,
        root: str | Path,
        subset: str,
        split: str,
        allow_partial: bool = False,
    ):
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        root = Path(root)
        source = "test" if split == "test" else "train"
        held_out = set(validation_ids(subset)) if source == "train" else set()

        self.subset = subset
        self.split = split
        self.groups: list[RowGroup] = []
        self.sample_ids: list[str] = []
        self._group_of: dict[str, int] = {}
        for path in _shard_files(root, subset, source, allow_partial):
            ids = pq.read_table(path, columns=["sample_id"])["sample_id"].to_pylist()
            meta = pq.read_metadata(path)
            start = 0
            for index in range(meta.num_row_groups):
                group_ids = ids[start : start + meta.row_group(index).num_rows]
                start += len(group_ids)
                if split == "train":
                    keep = tuple(i not in held_out for i in group_ids)
                elif split == "val":
                    keep = tuple(i in held_out for i in group_ids)
                else:
                    keep = (True,) * len(group_ids)
                if any(keep):
                    kept = [i for i, k in zip(group_ids, keep) if k]
                    self._group_of.update(dict.fromkeys(kept, len(self.groups)))
                    self.groups.append(RowGroup(str(path), index, keep))
                    self.sample_ids.extend(kept)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def sample(self, sample_id: str) -> Sample:
        """Random access to one sample (decodes its whole row group)."""
        if sample_id not in self._group_of:
            raise KeyError(f"{sample_id} is not in the {self.subset}/{self.split} split")
        group = self.groups[self._group_of[sample_id]]
        return next(s for s in read_row_group(group) if s.sample_id == sample_id)


def collate(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack samples into one block-row sparse LTM plus depth and occlusion maps.

    The LTMs of the batch are stacked vertically, giving a sparse matrix of shape
    (batch * camera_pixels, projector_pixels). Depth and occlusion are (batch, 1, H, W).
    """
    counts = [len(s.rows) for s in samples]
    indices = np.empty((2, sum(counts)), dtype=np.int64)
    np.concatenate([s.rows for s in samples], out=indices[0], casting="unsafe")
    indices[0] += np.repeat(np.arange(len(samples), dtype=np.int64) * PIXELS, counts)
    np.concatenate([s.cols for s in samples], out=indices[1], casting="unsafe")
    values = np.concatenate([s.values for s in samples]).astype(np.float32)
    values /= LTM_SCALE

    ltms = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        (len(samples) * PIXELS, PIXELS),
        is_coalesced=True,
    )
    depths = torch.from_numpy(np.stack([s.depth for s in samples]))[:, None]
    occlusions = torch.from_numpy(np.stack([s.occlusion for s in samples]))[:, None]
    return ltms, depths, occlusions


class LtmBatches(IterableDataset):
    """Iterates over collated batches of one split.

    Row groups are dealt out to distributed ranks and then to data-loader workers.
    With ``shuffle``, the row-group order is reshuffled every epoch (call
    ``set_epoch``) and samples are shuffled within ``buffer_row_groups`` row groups.
    With ``drop_last``, every rank yields the same number of full batches, which
    DistributedDataParallel needs. Without it, every sample is yielded exactly once.
    """

    def __init__(
        self,
        split: LtmSplit,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        buffer_row_groups: int = 4,
        seed: int = 0,
    ):
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.num_workers = max(num_workers, 1)
        self.buffer_row_groups = buffer_row_groups if shuffle else 1
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _assignments(self) -> list[list[list[int]]]:
        """Row-group indices for every [rank][worker]."""
        order = np.arange(len(self.split.groups))
        if self.shuffle:
            order = np.random.default_rng((self.seed, self.epoch)).permutation(order)
        per_rank = [order[r :: self.world_size] for r in range(self.world_size)]
        return [
            [per_rank[r][w :: self.num_workers].tolist() for w in range(self.num_workers)]
            for r in range(self.world_size)
        ]

    def _quotas(self, assignments) -> list[list[int]]:
        """Number of batches for every [rank][worker]."""
        groups = self.split.groups
        counts = [
            [sum(groups[g].num_samples for g in worker) for worker in rank] for rank in assignments
        ]
        if not self.drop_last:
            return [[-(-n // self.batch_size) for n in rank] for rank in counts]

        quotas = [[n // self.batch_size for n in rank] for rank in counts]
        common = min(sum(rank) for rank in quotas)
        for rank in quotas:
            while sum(rank) > common:
                rank[rank.index(max(rank))] -= 1
        return quotas

    def __len__(self) -> int:
        return sum(self._quotas(self._assignments())[self.rank])

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        if info is not None and info.num_workers != self.num_workers:
            raise ValueError("num_workers must match the DataLoader's num_workers")

        assignments = self._assignments()
        groups = assignments[self.rank][worker]
        quota = self._quotas(assignments)[self.rank][worker]
        rng = np.random.default_rng((self.seed, self.epoch, self.rank, worker))

        pending: list[Sample] = []
        emitted = 0
        for start in range(0, len(groups), self.buffer_row_groups):
            for g in groups[start : start + self.buffer_row_groups]:
                pending.extend(read_row_group(self.split.groups[g]))
            if self.shuffle:
                pending = [pending[i] for i in rng.permutation(len(pending))]
            while len(pending) >= self.batch_size and emitted < quota:
                yield collate(pending[: self.batch_size])
                del pending[: self.batch_size]
                emitted += 1
            if emitted == quota:
                return
        if pending and not self.drop_last:
            yield collate(pending)


def make_loader(
    split: LtmSplit,
    batch_size: int,
    *,
    train: bool,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    seed: int = 0,
) -> DataLoader:
    batches = LtmBatches(
        split,
        batch_size,
        shuffle=train,
        drop_last=train,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        seed=seed,
    )
    return DataLoader(
        batches,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def download(subset: str, root: str | Path, splits: tuple[str, ...] = ("train", "test")):
    from huggingface_hub import snapshot_download

    patterns = [f"{subset}/{split}-*.parquet" for split in splits]
    patterns += [f"metadata/{subset}/*", "README.md"]
    return snapshot_download(
        REPO_ID, repo_type="dataset", local_dir=str(root), allow_patterns=patterns
    )


def download_main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download the {REPO_ID} dataset from the Hugging Face Hub."
    )
    parser.add_argument("--subset", choices=SUBSETS + ("all",), default="all")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["train", "test"],
        help="Download only some splits, e.g. --splits test for evaluation",
    )
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "all" else (args.subset,)
    for subset in subsets:
        path = download(subset, args.root, tuple(args.splits))
        print(f"{subset}: {path}")
