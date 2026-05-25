"""Popularity recommender.

The "lazy" baseline every recsys paper has to beat. Ranks orgs by total donation
count in `train_donations`. Strong on metrics like NDCG@10 because popular orgs
*are* genuinely common in held-out test. Weak on catalog coverage and
diversity — exactly the failure modes we report on.
"""
from __future__ import annotations

import pandas as pd

from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self.ranked_orgs: list[str] = []

    def fit(self, *, train_donations: pd.DataFrame, orgs: pd.DataFrame, users: pd.DataFrame) -> None:
        if train_donations.empty:
            # Fall back to alphabetical so we always emit something.
            self.ranked_orgs = orgs["org_id"].tolist()
            return
        counts = train_donations["org_id"].value_counts()
        # Pad with any orgs that never appeared in train, in stable order.
        all_orgs = orgs["org_id"].tolist()
        seen = set(counts.index.tolist())
        unseen = [o for o in all_orgs if o not in seen]
        self.ranked_orgs = counts.index.tolist() + unseen

    def recommend(self, user_id: str, k: int) -> list[str]:
        return self.ranked_orgs[:k]
