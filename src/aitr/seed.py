"""Random-seed control for every script in this repository."""
import os
import random

import numpy as np


def best_device() -> str:
    """cuda (Colab) > mps (Apple GPU) > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    deterministic=True also asks PyTorch for deterministic kernels. GPU sampling
    is still not bit-identical across hardware, which is why raw generations are
    cached to results/raw/ and all metrics are computed from the cache.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
