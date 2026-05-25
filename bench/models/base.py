"""Recommender interface contract.

Every model in `bench/models/` implements this ABC. The orchestrator in
`bench/eval/run.py` only knows about `Recommender` — it never imports concrete
classes — so models stay substitutable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class Recommender(ABC):
    """A recommender that produces ranked org lists per user."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        *,
        train_donations: pd.DataFrame,
        orgs: pd.DataFrame,
        users: pd.DataFrame,
    ) -> None:
        """Fit the model on the train split.

        `train_donations` columns: `user_id`, `org_id`, `category`, `amount`, `timestamp_day`.
        `orgs` columns: `org_id`, `name`, `category`, `ntee_major`, `ntee_full`, `city`, `state`, `ein`.
        `users` columns: `user_id`, `primary_categories`, `verified_preference`, `avg_donation`, `is_category_locked`.
        """

    @abstractmethod
    def recommend(self, user_id: str, k: int) -> list[str]:
        """Return up to `k` org_ids ranked by predicted relevance.

        For users unseen at fit-time the model should still return a valid list
        (a cold-start strategy of the model's choice). May return fewer than
        `k` items if the corpus is exhausted.
        """

    def recommend_batch(self, user_ids: Sequence[str], k: int) -> dict[str, list[str]]:
        """Default O(U*K) loop. Models with cheaper batch inference should override."""
        return {u: self.recommend(u, k) for u in user_ids}
