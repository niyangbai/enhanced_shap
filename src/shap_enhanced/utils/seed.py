"""Set random seed for reproducibility."""

import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    :param int seed: Random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
