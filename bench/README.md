# Recommender benchmark — Slice 1

This directory is the *measured* slice of the donation-platform rebuild. It
replaces the previous `backend/ml/recommendation_engine.py` — which was a fake
that hard-coded "89% accuracy" and category-matched on random data — with a
real, reproducible benchmark of 6 models on a real nonprofit corpus.

> **One important note up front:** this work is a structural echo of patterns
> from production donation platforms; it is not a copy of any specific
> proprietary system. The architectural decisions are general ML-product
> patterns and the dataset is a sanitized real-org corpus + synthetic users.
> No real donor behavior is represented here.

## What gets measured

Six models on the same train/val/test split:

| Model | What it is |
|---|---|
| `random` | Sanity floor — recommendations chosen at random |
| `popularity` | Ranks by total donation count. The baseline every recsys has to beat. |
| `category-match` | Faithful port of the original code's logic — score = match-on-user's-top-category + popularity tie-break |
| `matrix-factorization` | Implicit-feedback ALS via the `implicit` library — classic non-neural baseline |
| `two-tower` | **Centerpiece.** PyTorch two-tower (user tower + org tower) with BPR loss, popularity-weighted negative sampling, FAISS top-K retrieval |
| `two-tower-content-init` | Ablation variant — org tower initialized from `sentence-transformers/all-MiniLM-L6-v2` embeddings of `f"{name} \| {category} \| {city}, {state}"`, then fine-tuned |

Each model is scored on:

- Ranking quality: NDCG@{5,10,20,50}, Recall@{5,10,20,50}, MRR, MAP@10, Precision@10
- Coverage: catalog coverage @ K=10, user coverage
- Diversity: mean intra-list category entropy
- Cold-start: NDCG@10 / Recall@10 on users with <3 train donations; recall on orgs with <5 train donations

Plus three **synthetic-user invariant tests** that gate the build:

1. **category-locked:** users whose train donations are 100% in one category should get top-10 mostly in that category
2. **diversity-floor:** multi-interest users (≥3 primary categories) should not get a 100%-single-category top-10
3. **beats-random:** the two-tower's NDCG@10 should be >2× random

## Quickstart

```bash
make bench-install       # one-time: pip install pinned deps into .venv-bench/
make bench               # full benchmark — writes bench/results/ (~15 min CPU)
make bench-fast          # smoke test on tiny data (~45 sec) — for CI
make bench-ablations     # 3×3×2 hyperparameter sweep (~1 hr)
make bench-test          # pytest unit tests for metrics
```

`make bench` writes:

- `bench/results/metrics.json` — machine-readable, every metric for every model
- `bench/results/REPORT.md` — human-readable, comparison table + invariant status
- `bench/results/training_curves.png` — two-tower BPR loss per epoch
- `bench/results/calibration.png` — predicted-score-decile vs observed-positive-rate
- `bench/results/comparison_table.png` — heatmap of the comparison table

## Reproducibility contract

- All randomness is seeded from `BENCH_SEED` (default `42`); change the env var to change the run
- The org corpus snapshot is **checked into the repo** at `bench/data/orgs.csv` so a clean checkout doesn't depend on ProPublica's API being up
- Dependencies are pinned in `bench/requirements.txt`
- `make bench` from a clean clone produces byte-identical `metrics.json` modulo `runtime_seconds` and timestamps on the same git SHA

## What this is, and what it isn't

- The **org corpus** is a sanitized snapshot of ProPublica's Nonprofit
  Explorer (https://projects.propublica.org/nonprofits/api). ~3K orgs sampled
  per-category from the full 5K snapshot for the headline run. The full CSV
  and the fetch script (`bench/scripts/fetch_propublica.py`) are in this repo.
- The **users and donations are synthetic**, generated from the seeded
  process in `bench/data/synthetic_*.py`. They are not real people and do
  not represent any real giving behavior, on any platform.
- All eval metrics are computed on the synthetic test split. They measure
  model quality *on this synthetic giving pattern.* They do **not** represent
  performance on any real production donation platform — including
  the deep-dive page's mention of Givelify, which is a structural reference
  only.
- The recommender's job here is: given a user's prior donation history
  (synthetic), produce a top-K list of orgs (real). The model never sees real
  user data and never produces real recommendations.

## Why this exists

The portfolio entry for this project previously claimed `1.5M+ users / 70K orgs
/ +25% retention / +35% speed`. None of those numbers were measured. This
benchmark replaces the headline stats with measured numbers — whatever they
turn out to be — and ships the code that produces them.

The deep-dive at `portfolioWebsite/src/work/donation-platform.md` describes
the full architectural picture (FastAPI gateway → recommender service →
Redis cache → FAISS → fallback paths). This `bench/` directory is **Slice 1**:
the recommender + offline eval. Slices 2–4 (gateway, cache, demo surface) are
explicitly out of scope here and will land in follow-up cycles.

## Layout

```
bench/
├── data/
│   ├── orgs.csv                      # ProPublica snapshot
│   ├── orgs_schema.md                # Columns + snapshot date
│   ├── synthetic_users.py            # Seeded user generator
│   └── synthetic_donations.py        # Seeded donation generator
├── models/
│   ├── base.py                       # Recommender ABC
│   ├── random_baseline.py
│   ├── popularity.py
│   ├── category_match.py
│   ├── matrix_factorization.py       # implicit ALS
│   ├── two_tower.py                  # Centerpiece (PyTorch + FAISS)
│   └── two_tower_content_init.py     # Sentence-transformer init variant
├── eval/
│   ├── metrics.py                    # NDCG, Recall, MRR, MAP, coverage, diversity
│   ├── invariants.py                 # Pass/fail synthetic-user tests
│   ├── calibration.py                # Decile vs observed-rate plot
│   ├── ablations.py                  # Hyperparameter sweep
│   └── run.py                        # Orchestrator
├── results/                          # Generated by `make bench` (gitignored)
├── scripts/
│   └── fetch_propublica.py           # Re-snapshot tool (manual, not in `make bench`)
├── tests/
│   └── test_metrics.py               # Hand-computed metric assertions
├── requirements.txt
└── README.md
```

The orchestrator at `eval/run.py` is the only file that imports across layers.
Everything else lives behind narrow interfaces:

- `data/` produces pandas DataFrames; knows nothing about models
- `models/*` all implement the `Recommender` ABC
- `eval/metrics.py` takes a `Recommender` + ground-truth dict; model-agnostic

That's it for Slice 1. Slice 2 (FastAPI gateway + fallback) and Slice 3
(Redis embedding cache + atomic invalidation) will wrap this benchmark's
trained two-tower into a live service.
