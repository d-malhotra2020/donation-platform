"""Implicit-feedback matrix factorization baseline.

Uses the `implicit` library's ALS. Treats donations as binary implicit feedback
weighted by donation count. This is the "non-neural" baseline that says
"do we even need a neural net?" — the two-tower has to beat *this* on the
maximalist eval, not just random or popularity.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender


class MatrixFactorizationRecommender(Recommender):
    name = "matrix-factorization"

    def __init__(self, factors: int = 32, iterations: int = 15, regularization: float = 0.01, seed: int = 42) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.seed = seed
        # Built at fit-time
        self.user_index: dict[str, int] = {}
        self.org_index: dict[str, int] = {}
        self.index_org: list[str] = []
        self.popularity_fallback: list[str] = []
        self._model = None
        self._user_items = None

    def fit(self, *, train_donations: pd.DataFrame, orgs: pd.DataFrame, users: pd.DataFrame) -> None:
        # implicit prints a thread warning on macOS; silence it for cleaner output.
        import os
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import implicit  # imported lazily so unit tests don't need the dep

        all_orgs = orgs["org_id"].tolist()
        all_users = users["user_id"].tolist()
        self.org_index = {o: i for i, o in enumerate(all_orgs)}
        self.user_index = {u: i for i, u in enumerate(all_users)}
        self.index_org = all_orgs

        if train_donations.empty:
            self.popularity_fallback = all_orgs
            return

        # Build user-item matrix.
        valid = train_donations[
            train_donations["user_id"].isin(self.user_index)
            & train_donations["org_id"].isin(self.org_index)
        ]
        if valid.empty:
            self.popularity_fallback = all_orgs
            return
        rows = valid["user_id"].map(self.user_index).to_numpy()
        cols = valid["org_id"].map(self.org_index).to_numpy()
        data = np.ones(len(valid), dtype=np.float32)
        self._user_items = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_index), len(self.org_index)),
        ).tocsr()
        # Sum duplicates so multiple donations from the same (user, org) accumulate.
        self._user_items.sum_duplicates()

        # Popularity ranking for cold-start fallback.
        self.popularity_fallback = (
            valid["org_id"].value_counts().index.tolist()
            + [o for o in all_orgs if o not in set(valid["org_id"])]
        )

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            use_gpu=False,
        )
        self._model.fit(self._user_items, show_progress=False)

    def recommend(self, user_id: str, k: int) -> list[str]:
        if self._model is None or user_id not in self.user_index:
            return self.popularity_fallback[:k]
        uidx = self.user_index[user_id]
        org_ids, _scores = self._model.recommend(
            uidx,
            self._user_items[uidx],
            N=k,
            filter_already_liked_items=False,
        )
        return [self.index_org[i] for i in org_ids]
