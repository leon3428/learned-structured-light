"""Single-GPU or ``torchrun`` multi-GPU setup."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Context:
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup() -> Context:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend, device_id=device if device.type == "cuda" else None)
    return Context(rank, world_size, device)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_buffers(module: torch.nn.Module) -> None:
    """Give every rank rank 0's buffers (BatchNorm statistics), as saved in checkpoints."""
    if dist.is_initialized():
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)
