import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Keep tests deterministic even when modules use global randomness."""
    np.random.seed(123)
    torch.manual_seed(123)
