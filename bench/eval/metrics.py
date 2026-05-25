"""Retrieval, coverage, and diversity metrics. No torch imports — pure numpy/python.

All metrics consume *ranked lists of item ids* and *sets of relevant ids*. The
recommender abstraction is purposely thin so metrics can be tested in isolation
against hand-computed expected values (see `bench/tests/test_metrics.py`).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized DCG with binary relevance. Standard form.

    Uses the position-discounted gain: gain at position i (1-indexed) is
    `rel_i / log2(i + 1)`. NDCG normalizes by the best achievable DCG (IDCG)
    where all relevant items appear in the top positions.
    """
    if not relevant:
        return 0.0
    k = max(0, k)
    dcg = 0.0
    for i, item in enumerate(ranking[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranking[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    k = max(1, k)
    hits = sum(1 for item in ranking[:k] if item in relevant)
    return hits / k


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for i, item in enumerate(ranking, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(ranking[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    denom = min(len(relevant), k)
    if denom == 0:
        return 0.0
    return score / denom


def aggregate_metrics(
    rankings: Mapping[str, Sequence[str]],
    truths: Mapping[str, set[str]],
    ks: Iterable[int] = (5, 10, 20, 50),
) -> dict[str, float]:
    """Compute per-user metrics + aggregate (mean over users)."""
    ks = list(ks)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def _bump(key: str, val: float) -> None:
        sums[key] = sums.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1

    for user_id, ranking in rankings.items():
        truth = truths.get(user_id, set())
        if not truth:
            continue
        for k in ks:
            _bump(f"ndcg@{k}", ndcg_at_k(ranking, truth, k))
            _bump(f"recall@{k}", recall_at_k(ranking, truth, k))
            _bump(f"precision@{k}", precision_at_k(ranking, truth, k))
        _bump("mrr", reciprocal_rank(ranking, truth))
        _bump("map@10", average_precision_at_k(ranking, truth, 10))

    return {k: sums[k] / counts[k] if counts[k] else 0.0 for k in sums}


def catalog_coverage(
    rankings: Mapping[str, Sequence[str]],
    catalog: Sequence[str],
    k: int,
) -> float:
    """Fraction of `catalog` that appears in at least one user's top-K."""
    if not catalog:
        return 0.0
    seen: set[str] = set()
    for ranking in rankings.values():
        for item in ranking[:k]:
            seen.add(item)
    return len(seen) / len(catalog)


def user_coverage(rankings: Mapping[str, Sequence[str]], users: Iterable[str], k: int) -> float:
    """Fraction of `users` who got at least one non-empty top-K recommendation."""
    users = list(users)
    if not users:
        return 0.0
    served = sum(1 for u in users if len(rankings.get(u, [])[:k]) > 0)
    return served / len(users)


def intra_list_category_entropy(categories: Sequence[str]) -> float:
    """Shannon entropy (base 2) of the category distribution in a single list."""
    if not categories:
        return 0.0
    counts = Counter(categories)
    n = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * math.log2(p)
    return H


def mean_intra_list_diversity(
    rankings: Mapping[str, Sequence[str]],
    item_to_category: Mapping[str, str],
    k: int,
) -> float:
    """Mean entropy across users of the top-K categorical distribution."""
    if not rankings:
        return 0.0
    entropies = []
    for ranking in rankings.values():
        cats = [item_to_category.get(item, "<unk>") for item in ranking[:k]]
        if cats:
            entropies.append(intra_list_category_entropy(cats))
    return sum(entropies) / len(entropies) if entropies else 0.0
