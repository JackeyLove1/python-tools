import torch
import torch.nn as nn
import logging
from typing import Optional
from logging.handlers import RotatingFileHandler


# To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py

def get_logger(filename: str, verbosity: int = 1, name: Optional[str] = "llama") -> logging.Logger:
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = RotatingFileHandler(filename, maxBytes=100 * 1024 * 1024, backupCount=10)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_lr(it: int):
    pass
