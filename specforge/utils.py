"""Small logging helpers shared by DLite training and inference."""

import logging

import torch.distributed as dist


logger = logging.getLogger(__name__)


def print_with_rank(message) -> None:
    if dist.is_available() and dist.is_initialized():
        logger.info("rank %s: %s", dist.get_rank(), message)
    else:
        logger.info("non-distributed: %s", message)


def print_on_rank0(message) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)


__all__ = ["print_on_rank0", "print_with_rank"]
