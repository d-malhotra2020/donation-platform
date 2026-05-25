"""Seeded synthetic donation-event generator.

Takes synthetic user profiles + the org corpus, samples donation events using
each user's `category_weights` as the ground-truth preference distribution
*plus* a latent (user, org) affinity score that gives the two-tower model
signal beyond category-matching alone.

The affinity component is deliberate: without it, `category-match` would be
the optimal model on this synthetic dataset by construction. With it, a
recommender that can learn org-level structure (matrix factorization,
two-tower) has something category-match cannot exploit.

Donation timestamps span a 12-month window so the chronological train/val/test
split in `bench/eval/run.py` produces non-degenerate folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic_users import UserProfile


def _build_latent_factors(
    user_ids: list[str],
    org_ids: list[str],
    n_factors: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a low-rank user/org factor decomposition.

    Each user has a `n_factors`-dim taste vector; each org has a `n_factors`-dim
    profile vector. Their dot product modulates donation probability *within*
    a chosen category — a real signal that a recommender with embedding
    capacity (two-tower, matrix factorization) can learn, but pure
    category-match cannot.
    """
    user_factors = rng.standard_normal((len(user_ids), n_factors)) * 0.5
    org_factors = rng.standard_normal((len(org_ids), n_factors)) * 0.5
    # Add a small org-level "appeal" intercept so popularity has a true signal too.
    org_appeal = rng.standard_normal(len(org_ids)) * 0.4
    org_factors = np.concatenate([org_factors, org_appeal[:, None]], axis=1)
    user_factors = np.concatenate([user_factors, np.ones((len(user_ids), 1))], axis=1)
    return user_factors, org_factors


def generate_donations(
    *,
    user_profiles: list[UserProfile],
    orgs: pd.DataFrame,
    seed: int,
    donations_per_user_min: int = 3,
    donations_per_user_max: int = 40,
    timestamp_days: int = 365,
    latent_factor_dim: int = 6,
    latent_strength: float = 1.0,
) -> pd.DataFrame:
    """Generate donation events.

    Returns a DataFrame with columns: `donation_id`, `user_id`, `org_id`,
    `category`, `amount`, `timestamp_day`.

    Process:
      1. For each user, sample a category from their `category_weights`.
      2. Within that category, sample an org weighted by softmax of
         `latent_strength * (user_factor . org_factor)`. This is the part
         category-match can't access.
    """
    rng = np.random.default_rng(seed)

    # Build the org-by-category index and orient factor arrays.
    orgs_indexed = orgs.reset_index(drop=True)
    org_id_to_row = {oid: i for i, oid in enumerate(orgs_indexed["org_id"].tolist())}
    by_category_idx: dict[str, np.ndarray] = {}
    for cat, group in orgs_indexed.groupby("category"):
        by_category_idx[cat] = np.array([org_id_to_row[o] for o in group["org_id"]])
    cats_arr = np.array(list(by_category_idx.keys()))

    user_ids = [p.user_id for p in user_profiles]
    user_factors, org_factors = _build_latent_factors(
        user_ids, orgs_indexed["org_id"].tolist(), latent_factor_dim, rng
    )
    user_row = {uid: i for i, uid in enumerate(user_ids)}

    rows: list[dict] = []
    donation_idx = 0
    for prof in user_profiles:
        # Each user draws a random number of donations in [min, max].
        # Category-locked users get *more* events so they have a clear signal in train.
        if prof.is_category_locked:
            lo = max(donations_per_user_min, 8)
            hi = max(donations_per_user_max, 25)
        else:
            lo = donations_per_user_min
            hi = donations_per_user_max
        n = int(rng.integers(lo, hi + 1))
        weights = np.array([prof.category_weights.get(c, 0.0) for c in cats_arr], dtype=float)
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()

        chosen_cats = rng.choice(cats_arr, size=n, p=weights)
        u_vec = user_factors[user_row[prof.user_id]]
        for cat in chosen_cats:
            org_idx_pool = by_category_idx.get(cat)
            if org_idx_pool is None or len(org_idx_pool) == 0:
                continue
            # Softmax-weighted sampling within category by latent affinity.
            scores = org_factors[org_idx_pool] @ u_vec * latent_strength
            scores = scores - scores.max()
            probs = np.exp(scores)
            probs = probs / probs.sum()
            chosen_org_idx = rng.choice(org_idx_pool, p=probs)
            org_id = orgs_indexed.iloc[int(chosen_org_idx)]["org_id"]
            amount = float(np.clip(rng.lognormal(mean=np.log(max(prof.avg_donation, 1.0)), sigma=0.5), 1.0, 5000.0))
            # Timestamp uniform across the window. Recommender split is chronological,
            # so this distribution drives which events end up in train vs test.
            ts_day = int(rng.integers(0, timestamp_days))
            rows.append({
                "donation_id": f"donation_{donation_idx:08d}",
                "user_id": prof.user_id,
                "org_id": org_id,
                "category": cat,
                "amount": amount,
                "timestamp_day": ts_day,
            })
            donation_idx += 1

    df = pd.DataFrame(rows)
    # Sort by timestamp so consumers can do a contiguous chronological split.
    df = df.sort_values("timestamp_day", kind="stable").reset_index(drop=True)
    return df


def split_chronological(
    donations: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split.

    For each user: last `test_frac` of their events → test, prior `val_frac` → val,
    rest → train. Filters out users with < 3 total donations (they can't yield a
    meaningful train/val/test split).
    """
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    for _, group in donations.groupby("user_id"):
        if len(group) < 3:
            train_rows.append(group)
            continue
        group = group.sort_values("timestamp_day", kind="stable")
        n = len(group)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - 1 - n_test)
        train_rows.append(group.iloc[: n - n_val - n_test])
        val_rows.append(group.iloc[n - n_val - n_test : n - n_test])
        test_rows.append(group.iloc[n - n_test :])
    train = pd.concat(train_rows, ignore_index=True) if train_rows else donations.iloc[:0]
    val = pd.concat(val_rows, ignore_index=True) if val_rows else donations.iloc[:0]
    test = pd.concat(test_rows, ignore_index=True) if test_rows else donations.iloc[:0]
    return train, val, test
