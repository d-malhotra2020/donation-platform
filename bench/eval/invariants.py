"""Synthetic-user invariant tests.

These are pass/fail assertions that gate the benchmark: if a recommender's
behavior on synthetic users with known preference profiles is wrong, the
build fails. The deep-dive page calls these out as more useful than coverage
metrics for an ML feature ("the bugs that matter are semantic, not syntactic").
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bench.data.synthetic_users import UserProfile
from bench.models.base import Recommender


@dataclass
class InvariantResult:
    name: str
    passed: bool
    score: float          # the measured value (e.g. fraction in-category)
    threshold: float      # the threshold needed to pass
    n_users: int          # users evaluated for this invariant
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "n_users": self.n_users,
            "notes": self.notes,
        }


def category_locked_invariant(
    recommender: Recommender,
    profiles: list[UserProfile],
    orgs: pd.DataFrame,
    k: int = 10,
    threshold: float = 0.6,  # 60% in-category — loose because synthetic donations don't fully constrain
) -> InvariantResult:
    """A user whose train donations are 100% in category C should get top-K mostly in C."""
    org_cat = orgs.set_index("org_id")["category"].astype(str).to_dict()
    locked_users = [p for p in profiles if p.is_category_locked]
    if not locked_users:
        return InvariantResult(
            name="category-locked",
            passed=True,
            score=1.0,
            threshold=threshold,
            n_users=0,
            notes="no category-locked synthetic users present",
        )
    fractions: list[float] = []
    for prof in locked_users:
        recs = recommender.recommend(prof.user_id, k)
        if not recs:
            continue
        target = prof.primary_categories[0]
        in_cat = sum(1 for o in recs if org_cat.get(o) == target)
        fractions.append(in_cat / len(recs))
    mean_frac = sum(fractions) / len(fractions) if fractions else 0.0
    return InvariantResult(
        name="category-locked",
        passed=mean_frac >= threshold,
        score=mean_frac,
        threshold=threshold,
        n_users=len(locked_users),
        notes=f"mean fraction of top-{k} in the user's locked category",
    )


def diversity_floor_invariant(
    recommender: Recommender,
    profiles: list[UserProfile],
    orgs: pd.DataFrame,
    k: int = 10,
    max_single_cat_frac: float = 0.95,
) -> InvariantResult:
    """Multi-interest users should not get a 100%-single-category top-K."""
    org_cat = orgs.set_index("org_id")["category"].astype(str).to_dict()
    multi = [p for p in profiles if not p.is_category_locked and len(p.primary_categories) >= 3]
    if not multi:
        return InvariantResult(
            name="diversity-floor",
            passed=True,
            score=0.0,
            threshold=max_single_cat_frac,
            n_users=0,
            notes="no multi-interest synthetic users with >=3 primary categories",
        )
    worst_max_frac = 0.0
    violations = 0
    for prof in multi:
        recs = recommender.recommend(prof.user_id, k)
        if not recs:
            continue
        cats = [org_cat.get(o, "<unk>") for o in recs]
        if not cats:
            continue
        single_max = max(cats.count(c) for c in set(cats)) / len(cats)
        worst_max_frac = max(worst_max_frac, single_max)
        if single_max > max_single_cat_frac:
            violations += 1
    return InvariantResult(
        name="diversity-floor",
        passed=violations == 0,
        score=worst_max_frac,
        threshold=max_single_cat_frac,
        n_users=len(multi),
        notes=f"worst single-category share of any user's top-{k}; threshold={max_single_cat_frac}",
    )


def beats_random_invariant(
    metrics_table: dict[str, dict[str, float]],
    metric: str = "ndcg@10",
) -> InvariantResult:
    """The two-tower must beat random on the headline metric (sanity check)."""
    rand = metrics_table.get("random", {}).get(metric)
    tt = None
    for name, row in metrics_table.items():
        if "two-tower" in name and "content" not in name:
            tt = row.get(metric)
            break
    if rand is None or tt is None:
        return InvariantResult(
            name="beats-random",
            passed=False,
            score=0.0,
            threshold=0.0,
            n_users=0,
            notes="random or two-tower not in metrics table",
        )
    return InvariantResult(
        name="beats-random",
        passed=tt > rand * 2.0,  # must be at least 2x random
        score=tt - rand,
        threshold=rand,
        n_users=0,
        notes=f"two-tower {metric}={tt:.4f} vs random {metric}={rand:.4f}; require >2x",
    )
