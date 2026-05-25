"""Seeded synthetic user generator.

Produces users with known preference profiles so the synthetic-user invariant
tests in `bench/eval/invariants.py` have a ground truth to assert against.

All randomness flows from a single `rng` so the same `seed` produces byte-identical
output. Tests rely on this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class UserProfile:
    """The fields each synthetic user carries.

    A user's `category_weights` is the ground-truth distribution we sample
    donations from. The invariant tests check that the recommender's top-K
    aligns with these weights.
    """

    user_id: str
    primary_categories: tuple[str, ...]
    category_weights: dict[str, float]
    verified_preference: float  # P(donate to verified org), if such a field existed
    avg_donation: float
    is_category_locked: bool  # True if user is single-category for invariants


def generate_users(
    *,
    n_users: int,
    categories: list[str],
    seed: int,
    fraction_category_locked: float = 0.05,
    fraction_multi_interest: float = 0.55,
    fraction_eclectic: float = 0.40,
) -> tuple[pd.DataFrame, list[UserProfile]]:
    """Generate synthetic users.

    Three profile types coexist:
      - category-locked: 100% weight on a single category. Used by the
        category-lock invariant test.
      - multi-interest: weight concentrated on 2-3 categories.
      - eclectic: uniform-ish weights across many categories.
    """
    if abs(fraction_category_locked + fraction_multi_interest + fraction_eclectic - 1.0) > 1e-6:
        raise ValueError("user-profile fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    profiles: list[UserProfile] = []
    rows: list[dict] = []

    n_locked = int(round(n_users * fraction_category_locked))
    n_multi = int(round(n_users * fraction_multi_interest))
    n_eclectic = n_users - n_locked - n_multi

    def _make_user(idx: int, profile_type: str) -> UserProfile:
        user_id = f"user_{idx:07d}"
        cats = list(categories)
        if profile_type == "locked":
            primary = (rng.choice(cats),)
            weights = {c: 0.0 for c in cats}
            weights[primary[0]] = 1.0
        elif profile_type == "multi":
            k = int(rng.integers(2, 4))  # 2 or 3
            primary = tuple(rng.choice(cats, size=k, replace=False).tolist())
            raw = rng.dirichlet(np.ones(k) * 2.0)  # concentrated
            weights = {c: 0.0 for c in cats}
            for c, w in zip(primary, raw):
                weights[c] = float(w)
        else:  # eclectic
            k = int(rng.integers(5, len(cats)))
            primary = tuple(rng.choice(cats, size=k, replace=False).tolist())
            raw = rng.dirichlet(np.ones(k) * 0.5)  # spread out
            weights = {c: 0.0 for c in cats}
            for c, w in zip(primary, raw):
                weights[c] = float(w)
        # Donation amount: lognormal mean ~$50
        avg = float(np.clip(rng.lognormal(mean=3.5, sigma=0.7), 5.0, 1500.0))
        verified_pref = float(rng.beta(2.0, 2.0))
        prof = UserProfile(
            user_id=user_id,
            primary_categories=primary,
            category_weights=weights,
            verified_preference=verified_pref,
            avg_donation=avg,
            is_category_locked=(profile_type == "locked"),
        )
        return prof

    idx = 0
    for _ in range(n_locked):
        profiles.append(_make_user(idx, "locked"))
        idx += 1
    for _ in range(n_multi):
        profiles.append(_make_user(idx, "multi"))
        idx += 1
    for _ in range(n_eclectic):
        profiles.append(_make_user(idx, "eclectic"))
        idx += 1

    # Shuffle so consumers can't trivially guess profile from order.
    rng.shuffle(profiles)

    for prof in profiles:
        rows.append({
            "user_id": prof.user_id,
            "primary_categories": "|".join(prof.primary_categories),
            "verified_preference": prof.verified_preference,
            "avg_donation": prof.avg_donation,
            "is_category_locked": prof.is_category_locked,
        })

    return pd.DataFrame(rows), profiles
