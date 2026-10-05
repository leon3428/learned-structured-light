"""Train a decoder with learned or fixed projection patterns.

Single GPU:   sl-train configs/diffuse_learned_3.toml
Two GPUs:     torchrun --nproc-per-node 2 -m learned_sl.train configs/diffuse_learned_3.toml

Each run writes ``<out-dir>/<name>/`` with the config, ``metrics.jsonl`` (one line per
epoch), ``last.pth`` and ``best.pth`` (lowest validation MAE, used for testing).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from learned_sl import distributed
from learned_sl.config import Config, build_model, load_config
from learned_sl.data import DEFAULT_ROOT, LtmSplit, make_loader
from learned_sl.evaluate import evaluate
from learned_sl.losses import SSIM, masked_rmse, masked_ssim_loss, pattern_variance_loss


def make_optimizers(config: Config, model) -> list[torch.optim.Optimizer]:
    decoder = torch.optim.Adam(
        model.decoder.parameters(), lr=config.decoder_lr, weight_decay=config.weight_decay
    )
    if not config.learn_patterns:
        return [decoder]
    patterns = torch.optim.Adam([model.projection_matrix], lr=config.pattern_lr)
    return [patterns, decoder]


def train(
    config: Config,
    data_root: str,
    out_dir: Path,
    num_workers: int = 4,
    max_batches: int | None = None,
    allow_partial: bool = False,
    wandb_project: str | None = None,
) -> None:
    ctx = distributed.setup()
    if config.batch_size % ctx.world_size != 0:
        raise ValueError("batch_size must be divisible by the number of GPUs")
    batch_size = config.batch_size // ctx.world_size
    torch.manual_seed(config.seed)

    train_split = LtmSplit(data_root, config.subset, "train", allow_partial)
    val_split = LtmSplit(data_root, config.subset, "val", allow_partial)
    train_loader = make_loader(
        train_split,
        batch_size,
        train=True,
        rank=ctx.rank,
        world_size=ctx.world_size,
        num_workers=num_workers,
        seed=config.seed,
    )
    val_loader = make_loader(
        val_split,
        batch_size,
        train=False,
        rank=ctx.rank,
        world_size=ctx.world_size,
        num_workers=num_workers,
    )

    model = build_model(config).to(ctx.device)
    ddp_model = (
        DistributedDataParallel(model, device_ids=[ctx.device]) if ctx.world_size > 1 else model
    )
    optimizers = make_optimizers(config, model)
    ssim_fn = SSIM(val_range=1.0).to(ctx.device)

    run_dir = out_dir / config.name
    run = None
    if ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.toml").write_text(config.to_toml())
        (run_dir / "metrics.jsonl").write_text("")
        if wandb_project is not None:
            import wandb

            run = wandb.init(project=wandb_project, name=config.name, config=vars(config))
        print(
            f"{config.name}: {len(train_split)} train / {len(val_split)} val samples, "
            f"{ctx.world_size} GPU(s), {batch_size} samples per GPU"
        )

    best_mae = float("inf")
    for epoch in range(1, config.epochs + 1):
        ddp_model.train()
        train_loader.dataset.set_epoch(epoch)
        started = time.time()
        total = len(train_loader) if max_batches is None else min(max_batches, len(train_loader))
        batches = tqdm(train_loader, total=total, desc=f"epoch {epoch}", disable=not ctx.is_main)
        sums = torch.zeros(4, device=ctx.device)
        for step, (ltms, depths, occlusions) in enumerate(batches):
            if step == total:
                break
            ltms = ltms.to(ctx.device, non_blocking=True)
            depths = depths.to(ctx.device, non_blocking=True)
            occlusions = occlusions.to(ctx.device, non_blocking=True)

            outputs, patterns = ddp_model(ltms)
            outputs = outputs.float()

            rmse = masked_rmse(outputs, depths, occlusions)
            loss = rmse
            ssim_term = masked_ssim_loss(ssim_fn, outputs, depths, occlusions)
            if config.ssim_weight:
                loss = loss + config.ssim_weight * ssim_term
            variance_term = pattern_variance_loss(patterns)
            if config.learn_patterns and config.variance_weight:
                loss = loss + config.variance_weight * variance_term

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            for optimizer in optimizers:
                optimizer.step()

            sums += torch.stack(
                [
                    loss.detach(),
                    rmse.detach(),
                    ssim_term.detach(),
                    torch.ones((), device=ctx.device),
                ]
            )
            if run is not None and step % 10 == 0:
                run.log(
                    {
                        "train/loss": loss.item(),
                        "train/rmse": rmse.item(),
                        "train/ssim_loss": ssim_term.item(),
                        "train/pattern_variance_loss": variance_term.item(),
                        "train/epoch": epoch - 1 + step / total,
                    }
                )

        distributed.broadcast_buffers(model)
        val = evaluate(model, val_loader, ctx, max_batches=max_batches)
        if ctx.is_main:
            loss_sum, rmse_sum, ssim_sum, steps = sums.tolist()
            record = {
                "epoch": epoch,
                "train_loss": loss_sum / steps,
                "train_rmse_m": rmse_sum / steps,
                "train_ssim_loss": ssim_sum / steps,
                **{f"val_{k}": v for k, v in val.items()},
                "seconds": time.time() - started,
            }
            print(json.dumps(record))
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            if run is not None:
                run.log({f"val/{k}": v for k, v in val.items()} | {"epoch": epoch})

            torch.save(model.state_dict(), run_dir / "last.pth")
            if val["mae_mm"] < best_mae:
                best_mae = val["mae_mm"]
                torch.save(model.state_dict(), run_dir / "best.pth")
                (run_dir / "best.json").write_text(json.dumps(record, indent=2) + "\n")
        distributed.barrier()

    if run is not None:
        run.finish()
    distributed.teardown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--num-workers", type=int, default=4, help="data workers per GPU")
    parser.add_argument("--wandb-project", default=None, help="also log to Weights & Biases")
    parser.add_argument("--epochs", type=int, default=None, help="override for quick checks")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="limit train and validation batches per epoch (for quick checks)",
    )
    parser.add_argument(
        "--allow-partial-dataset",
        action="store_true",
        help="allow missing shards (for quick checks; results will not match the paper)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config = Config(**(vars(config) | {"epochs": args.epochs}))
    train(
        config,
        args.data_root,
        args.out_dir,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        allow_partial=args.allow_partial_dataset,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
