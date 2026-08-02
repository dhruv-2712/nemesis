"""Feature bank — real transaction feature vectors for seeding sandbox nodes.

A user-built sandbox pattern has topology but no transaction features, and the
GNN needs 165 features per node to produce a risk score. This module holds a
small, committed bank of *real* feature vectors sampled from the Elliptic graph
(illicit and licit), so the sandbox can seed each node from the chosen profile
without shipping the full 135 MB graph. Seeding is honest-by-label: the UI states
that features are sampled from the illicit/licit distribution.

Build once (needs the graph + checkpoint):  python -m backend.models.feature_bank
Then it loads from the committed npz at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BANK_PATH = Path("backend/models/feature_bank.npz")
PER_CLASS = 400  # illicit + licit vectors stored


def build_bank(seed: int = 42) -> None:
    """Sample real illicit/licit feature rows from the graph into a committed npz."""
    import torch

    from backend.reasoning.cluster import illicit_probs

    data = torch.load("data/graphs/elliptic.pt", weights_only=False)
    probs = illicit_probs(data)

    rng = np.random.default_rng(seed)
    # illicit: real illicit label AND scored illicit (coherent illicit profile)
    illicit_idx = ((data.y == 1) & (probs >= 0.9)).nonzero(as_tuple=True)[0].numpy()
    licit_idx = ((data.y == 0) & (probs <= 0.1)).nonzero(as_tuple=True)[0].numpy()
    illicit_pick = rng.choice(illicit_idx, size=min(PER_CLASS, len(illicit_idx)), replace=False)
    licit_pick = rng.choice(licit_idx, size=min(PER_CLASS, len(licit_idx)), replace=False)

    # Store the full 166-dim rows (time_step col kept; model_features drops it later).
    illicit = data.x[illicit_pick].numpy().astype(np.float32)
    licit = data.x[licit_pick].numpy().astype(np.float32)
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BANK_PATH, illicit=illicit, licit=licit)
    print(f"feature bank -> {BANK_PATH}  (illicit {illicit.shape}, licit {licit.shape})")


_BANK: dict | None = None


def _load() -> dict:
    global _BANK
    if _BANK is None:
        with np.load(BANK_PATH) as npz:
            _BANK = {"illicit": npz["illicit"], "licit": npz["licit"]}
    return _BANK


def sample_features(profile: str, n: int, seed: int | None = None) -> np.ndarray:
    """Return (n, 166) feature rows sampled from the illicit or licit bank."""
    bank = _load()
    pool = bank["illicit"] if profile == "illicit" else bank["licit"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pool), size=n)
    return pool[idx].copy()


if __name__ == "__main__":
    build_bank()
