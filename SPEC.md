# Donation Platform — Recommender + Offline Eval (Slice 1)

**Status:** Approved, in implementation
**Designed:** 2026-05-24
**Slice scope:** Recommender service + offline evaluation. Out of scope: live API gateway (Slice 2), Redis embedding cache (Slice 3), web dashboard (Slice 4), mobile client, payments, geocode.

## Why this exists

This repo previously claimed `1.5M+ users / 70K orgs / 25% retention / 35% speed` on the portfolio. None of those numbers were measured — the existing code is a fake FastAPI surface with random data and a category-match recommender masquerading as ML. Slice 1 replaces the fabricated claims with a real, measurable, reproducible benchmark.

The deep-dive page at `portfolioWebsite/src/work/donation-platform.md` describes an aspirational architecture (FastAPI gateway → separate recommender service → FAISS → Redis cache → fallback paths). The architecture is honest as a *design narrative*; the prior code did not honor it. This slice ships the recommender + offline eval that the architecture is built around. Later slices fill in the operational surface.

## What ships in Slice 1

A self-contained `bench/` tree at the repo root with:
- Real org corpus: ~5K nonprofit snapshot from ProPublica Nonprofit Explorer, checked into the repo
- Synthetic users + donations: seeded generators producing reproducible event streams with known preference profiles
- 5 baseline models: random / popularity / category-match (legacy code's logic) / matrix factorization (`implicit` ALS) / two-tower neural net (PyTorch + FAISS top-K)
- 1 ablation variant: two-tower with sentence-transformer org-tower initialization
- Maximalist eval bundle: NDCG@{5,10,20}, Recall@{5,10,20,50}, MRR, MAP@10, Precision@10, catalog coverage, intra-list diversity, cold-user / cold-org slices, calibration plot, training curves, ablation sweep
- Pass/fail synthetic-user invariant tests: category-locked / verified-org-preference / diversity-floor
- Reproducibility: `make bench` from clean checkout, fixed seeds, bit-identical results per git SHA
- `bench/results/REPORT.md` and `bench/results/metrics.json` as portfolio-consumable artifacts

## Architecture & repo layout

```
donation-platform/
├── bench/
│   ├── README.md                     # Reproducibility + honesty footer
│   ├── requirements.txt              # Pinned deps
│   ├── data/
│   │   ├── orgs.csv                  # ProPublica snapshot (~5K rows)
│   │   ├── orgs_schema.md            # Columns + snapshot date
│   │   ├── synthetic_users.py        # Seeded user generator
│   │   └── synthetic_donations.py    # Seeded donation generator
│   ├── models/
│   │   ├── base.py                   # Recommender ABC
│   │   ├── random_baseline.py
│   │   ├── popularity.py
│   │   ├── category_match.py
│   │   ├── matrix_factorization.py   # `implicit` ALS
│   │   ├── two_tower.py              # Centerpiece (PyTorch)
│   │   └── two_tower_content_init.py # Ablation variant
│   ├── eval/
│   │   ├── metrics.py                # NDCG, Recall, MRR, MAP, coverage, diversity
│   │   ├── invariants.py             # Synthetic-user assertions
│   │   ├── calibration.py
│   │   ├── ablations.py              # 3x3x2 grid sweep
│   │   └── run.py                    # Orchestrator
│   ├── results/                      # Gitignored except .gitkeep
│   └── scripts/
│       └── fetch_propublica.py       # Manual re-snapshot tool
├── Makefile                          # bench / bench-fast / bench-ablations / bench-clean
└── backend/                          # Legacy fake API — replaced in Slice 2
```

**Module isolation contract:**
- `bench/data/` knows nothing about models. Output: pandas DataFrames + a `train/val/test` split function.
- `bench/models/*.py` all implement the same `Recommender` ABC. Substitutable, comparable.
- `bench/eval/metrics.py` takes any `Recommender` + held-out test set, returns a metrics dict. Model-agnostic.
- `bench/eval/run.py` is the only thing that knows about all three layers.

## Two-tower model

**User tower:** `nn.Embedding(num_users, embed_dim)` → 2-layer MLP with ReLU + dropout → L2-normalized output. `embed_dim ∈ {16, 32, 64}`, default 32.

**Org tower (default):** `nn.Embedding(num_orgs, embed_dim)` + one-hot category embedding concatenated → MLP → L2-normalized.

**Org tower (content-init variant):** start from `sentence-transformers/all-MiniLM-L6-v2` embeddings of `f"{name} | {category} | {description}"`, fine-tuned on interactions. This is the variant for cold-start eval.

**Training:** BPR (Bayesian Personalized Ranking) loss with sigmoid + popularity-weighted negative sampling. Adam, lr=1e-3, weight_decay=1e-5, batch=1024, in-batch + 5 sampled negatives per positive. Early stop on val NDCG@10 with patience=3. 20–50 epochs, < 5 min CPU at chosen scale.

**Inference:** precompute all org embeddings → FAISS `IndexFlatIP` → `recommend(user_id, K)` runs user-tower forward → FAISS query → top-K org ids.

## Eval bundle (maximalist)

**Data split:** chronological by donation timestamp — last 20% of events per user → test, prior 20% → val, rest → train. Chronological prevents look-ahead leakage.

**Headline retrieval (every model):**
- NDCG@{5, 10, 20} — NDCG@10 is portfolio headline
- Recall@{5, 10, 20, 50}
- MRR
- MAP@10
- Precision@10

**Coverage + diversity (centerpiece + popularity baseline):**
- Catalog coverage @ K=10
- User coverage
- Intra-list category entropy
- Intra-list pairwise cosine distance

**Cold-start slice:**
- Cold-user NDCG@10 / Recall@10 on users with <3 train donations
- Cold-org Recall@10 on orgs with <5 train donations (where content-init should shine)

**Synthetic-user invariants (pass/fail, gates the build):**
- Category-locked user → top-10 must be ≥80% in category
- Verified-org preference user → top-10 must be ≥70% verified
- Multi-interest user → top-10 must not be 100% single-category

**Calibration plot:** centerpiece only. Bin by predicted score decile, plot observed positive rate.

**Training curves:** train/val NDCG@10 logged per epoch for both two-tower variants.

**Ablations (separate target `make bench-ablations`):** `embed_dim ∈ {16,32,64}` × `negative_samples ∈ {1,5,10}` × `with_content_init ∈ {true,false}` = 18 runs, ~1 hr.

## Reproducibility

- All randomness seeded (numpy, torch, random) via single `BENCH_SEED=42`
- Pinned deps in `bench/requirements.txt`
- `make bench` from clean checkout: pip install requirements → `python -m bench.eval.run`
- Results stamped with `git_sha`, `dataset_snapshot_date`, `runtime_seconds`, `seeds` in `metrics.json`
- Smoke test `make bench-fast`: 50 orgs, 200 users, 1 epoch, asserts metric ordering. <30 sec.
- Reproducibility test: `make bench-fast` twice → diff metrics.json → byte-identical except runtime_seconds.

## Portfolio synchronization

**`portfolioWebsite/src/components/Projects.jsx`** — donation-platform card. Replace fabricated stats:
```js
// Before:
stats: [
  { lbl: 'Users', val: '1.5M+' },
  { lbl: 'Orgs', val: '70K+' },
  { lbl: 'Retention', val: '+25%' },
  { lbl: 'Speed', val: '+35%' }
]
// After (numbers from bench/results/metrics.json):
stats: [
  { lbl: 'NDCG@10', val: '0.XX' },
  { lbl: 'vs popularity', val: '+XX%' },
  { lbl: 'Orgs', val: '5K real' },
  { lbl: 'Reproducible', val: 'make bench' }
]
```

**`portfolioWebsite/src/work/donation-platform.md`** — append "Measured results" section linking to `bench/results/REPORT.md` with a 3-row headline table.

If the two-tower fails to beat popularity, we report that honestly.

## Honesty contract (in `bench/README.md`)

- Org corpus = sanitized ProPublica Nonprofit Explorer snapshot from {date}; ~5K orgs sampled
- Users + donations are synthetic, generated from seeded process in `bench/data/synthetic_*.py`. Not real people.
- All eval metrics on synthetic test split. Measure model quality *on this synthetic giving pattern.* Do not represent Givelify or any real production system.
- Reproducibility: `make bench` from clean checkout, fixed seeds, bit-identical results.

## Testing strategy

- Unit tests for `bench/eval/metrics.py` — pytest with hand-crafted rankings whose NDCG/Recall/MRR are computed by hand
- Smoke test `make bench-fast` asserts metric ordering (random < popularity < two-tower)
- Invariant tests double as functional tests in `make bench`
- Reproducibility test: `make bench-fast` × 2 → byte-identical metrics.json

## Explicit out-of-scope

- FastAPI gateway / live serving (Slice 2)
- Redis embedding cache + `(entity_type, id, model_version)` keying (Slice 3)
- Web dashboard / live demo (Slice 4)
- Mobile React Native client, Stripe payments, geocode, multi-region
- Online learning / real-time updates (deep-dive rejects this)
- ProPublica re-snapshot cron (`fetch_propublica.py` is a manual tool)
- GPU — `make bench` is CPU-only by contract
