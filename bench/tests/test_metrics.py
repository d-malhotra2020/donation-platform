"""Hand-computed metric tests. No torch imports.

Each test uses a tiny ranking whose expected metric value is computed by hand,
so a bug in the metric implementation can't hide behind a "magic constant from
some other library."
"""
from __future__ import annotations

import math

import pytest

from bench.eval import metrics


def test_ndcg_perfect_ranking_is_one():
    # All relevant items appear at the top → NDCG = 1.0
    recs = ["a", "b", "c", "d"]
    relevant = {"a", "b"}
    assert metrics.ndcg_at_k(recs, relevant, k=4) == pytest.approx(1.0)


def test_ndcg_handcomputed():
    # Recs: [a, b, c, d], relevant: {b, d}
    # DCG = 1/log2(2+1) + 1/log2(4+1) = 1/log2(3) + 1/log2(5)
    # IDCG = 1/log2(2) + 1/log2(3)  (two relevant items would ideally be at positions 1, 2)
    recs = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    assert metrics.ndcg_at_k(recs, relevant, k=4) == pytest.approx(expected)


def test_ndcg_no_relevant_is_zero():
    recs = ["a", "b", "c"]
    assert metrics.ndcg_at_k(recs, set(), k=3) == 0.0


def test_recall_at_k():
    recs = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "z"}  # z is relevant but not in any K
    assert metrics.recall_at_k(recs, relevant, k=2) == pytest.approx(1 / 3)
    assert metrics.recall_at_k(recs, relevant, k=3) == pytest.approx(2 / 3)
    assert metrics.recall_at_k(recs, relevant, k=5) == pytest.approx(2 / 3)


def test_recall_no_relevant_is_zero():
    assert metrics.recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_precision_at_k():
    recs = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert metrics.precision_at_k(recs, relevant, k=4) == pytest.approx(0.5)
    assert metrics.precision_at_k(recs, relevant, k=2) == pytest.approx(0.5)
    assert metrics.precision_at_k(recs, relevant, k=1) == pytest.approx(1.0)


def test_mrr_first_relevant_at_position_2():
    recs = ["a", "b", "c"]
    relevant = {"b"}
    assert metrics.reciprocal_rank(recs, relevant) == pytest.approx(0.5)


def test_mrr_no_relevant():
    assert metrics.reciprocal_rank(["a"], set()) == 0.0


def test_map_at_k_handcomputed():
    # Recs: [a, b, c, d], relevant: {a, c}
    # AP = (1/1 + 2/3) / 2 = (1 + 0.6667) / 2 = 0.8333...
    recs = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert metrics.average_precision_at_k(recs, relevant, k=4) == pytest.approx(expected)


def test_aggregate_passes_through_per_user():
    rankings = {
        "u1": ["a", "b", "c"],
        "u2": ["x", "y", "z"],
    }
    truths = {"u1": {"a"}, "u2": {"y"}}
    agg = metrics.aggregate_metrics(rankings, truths, ks=[1, 3])
    # u1 NDCG@1 = 1.0, u2 NDCG@1 = 0
    assert agg["ndcg@1"] == pytest.approx(0.5)
    # u1 Recall@3 = 1.0, u2 Recall@3 = 1.0
    assert agg["recall@3"] == pytest.approx(1.0)


def test_catalog_coverage():
    rankings = {
        "u1": ["a", "b"],
        "u2": ["b", "c"],
    }
    catalog = ["a", "b", "c", "d", "e"]  # only 3 of 5 ever recommended
    assert metrics.catalog_coverage(rankings, catalog, k=2) == pytest.approx(3 / 5)


def test_intra_list_category_entropy_uniform_high():
    # 4 distinct categories → entropy should equal log2(4) = 2.0
    cats = ["A", "B", "C", "D"]
    expected = 2.0
    assert metrics.intra_list_category_entropy(cats) == pytest.approx(expected)


def test_intra_list_category_entropy_all_same_is_zero():
    cats = ["A", "A", "A", "A"]
    assert metrics.intra_list_category_entropy(cats) == 0.0
