"""Category-match recommender.

A faithful port of the logic in the original `backend/ml/recommendation_engine.py`:

    score = 0.5 * (org.category in user.interests) + ratings/rating boosts

Rather than fake "interests", we infer user category preferences from train
donation counts. For each user, rank orgs by:
  - 1.0 if org's category is the user's #1 donated-to category in train
  - 0.5 if it appears in their top-3
  - 0.0 otherwise
Then break ties by org popularity (matches the original's "rating" stand-in).

This is the baseline that says: "you don't need ML, just match categories."
The two-tower has to beat *this*, not just random.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .base import Recommender


class CategoryMatchRecommender(Recommender):
    name = "category-match"

    def __init__(self) -> None:
        self.user_top_categories: dict[str, list[str]] = {}
        self.global_top_categories: list[str] = []
        self.orgs_by_category: dict[str, list[str]] = {}
        self.popularity_rank: dict[str, int] = {}
        self.all_orgs: list[str] = []

    def fit(self, *, train_donations: pd.DataFrame, orgs: pd.DataFrame, users: pd.DataFrame) -> None:
        # Per-user top categories (stable ranking)
        self.user_top_categories = {}
        if not train_donations.empty:
            for user_id, group in train_donations.groupby("user_id"):
                counts = group["category"].value_counts()
                self.user_top_categories[user_id] = counts.index.tolist()
            # Global ranking for cold-start fallback
            self.global_top_categories = (
                train_donations["category"].value_counts().index.tolist()
            )
        else:
            self.global_top_categories = list(orgs["category"].unique())

        # Pre-bucket orgs by category, sorted by popularity
        pop = train_donations["org_id"].value_counts() if not train_donations.empty else pd.Series(dtype=int)
        self.popularity_rank = {oid: -int(c) for oid, c in pop.items()}
        all_orgs_list: list[str] = []
        self.orgs_by_category = {}
        for cat, group in orgs.groupby("category"):
            sorted_orgs = sorted(
                group["org_id"].tolist(),
                key=lambda o: (self.popularity_rank.get(o, 0), o),
            )
            self.orgs_by_category[cat] = sorted_orgs
            all_orgs_list.extend(sorted_orgs)
        self.all_orgs = all_orgs_list

    def recommend(self, user_id: str, k: int) -> list[str]:
        user_cats = self.user_top_categories.get(user_id) or self.global_top_categories
        recs: list[str] = []
        seen: set[str] = set()
        # Walk down the user's category ranking, pulling the top-popularity orgs from each.
        for cat in user_cats:
            for org_id in self.orgs_by_category.get(cat, []):
                if org_id not in seen:
                    recs.append(org_id)
                    seen.add(org_id)
                    if len(recs) >= k:
                        return recs
        # Pad with overall-popular orgs if we still need more.
        for org_id in self.all_orgs:
            if org_id not in seen:
                recs.append(org_id)
                seen.add(org_id)
                if len(recs) >= k:
                    return recs
        return recs
