"""Shared distributed training helpers."""

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(strict: bool = False, launch_hint: str | None = None) -> tuple[int, int, int]:
    """
    Initialize torch.distributed from torchrun environment variables.

    Returns:
        (rank, world_size, local_rank)
    """
    has_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
    if not has_env:
        if strict:
            hint = launch_hint or "torchrun --standalone --nproc_per_node=4 <script.py> --strategy ddp ..."
            raise RuntimeError(f"DDP 需要用 torchrun 启动，例如：\n  {hint}")
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank
