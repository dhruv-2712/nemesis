"""Temporal train/val/test masks for the Elliptic graph.

Elliptic has 49 discrete time steps. The canonical, leakage-free evaluation
(Weber et al. 2019) trains on early steps and tests on later ones, since a real
detector only ever sees the past. We reserve a small validation band between
train and test for early stopping / model selection.

  train:  time steps 1..29   (model selection: fit)
  val:    time steps 30..34   (early stopping)
  test:   time steps 35..49   (held-out, reported metrics)

Only *labeled* nodes (y in {0, 1}) contribute to any mask — the 157k unknown
nodes still participate in message passing (they carry structure) but are never
used for loss or metrics.
"""

import torch
from torch import Tensor

from backend.graph.schema import TIME_STEP_IDX

TRAIN_MAX_STEP = 29
VAL_MAX_STEP = 34  # test is everything after this, up to 49


def temporal_masks(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return boolean (train_mask, val_mask, test_mask) over labeled nodes."""
    time_step = x[:, TIME_STEP_IDX]
    labeled = y >= 0

    train_mask = labeled & (time_step <= TRAIN_MAX_STEP)
    val_mask = labeled & (time_step > TRAIN_MAX_STEP) & (time_step <= VAL_MAX_STEP)
    test_mask = labeled & (time_step > VAL_MAX_STEP)
    return train_mask, val_mask, test_mask


def model_features(x: Tensor) -> Tensor:
    """Drop the time_step column (index 0); return the 165 learnable features."""
    return x[:, TIME_STEP_IDX + 1 :]
