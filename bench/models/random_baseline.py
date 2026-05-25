"""Random recommender. Sanity-floor baseline: NDCG@10 should be near zero."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.orgs: np.ndarray | None = None

    def fit(self, *, train_donations: pd.DataFrame, orgs: pd.DataFrame, users: pd.DataFrame) -> None:
        self.orgs = orgs["org_id"].to_numpy()

    def recommend(self, user_id: str, k: int) -> list[str]:
        assert self.orgs is not None
        # Per-user RNG seeded by user_id hash so output is deterministic given seed.
        h = (hash(user_id) ^ self.seed) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        idx = rng.choice(len(self.orgs), size=min(k, len(self.orgs)), replace=False)
        return [str(o) for o in self.orgs[idx]]
