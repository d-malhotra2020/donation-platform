"""Benchmark orchestrator.

The only file in `bench/` that imports from all three layers (data, models, eval).
Loads the org snapshot, generates synthetic users + donations, splits
chronologically, fits all baselines + centerpiece, computes the full eval bundle,
and writes:

    bench/results/metrics.json     # machine-readable
    bench/results/REPORT.md        # human-readable
    bench/results/training_curves.png
    bench/results/calibration.png
    bench/results/comparison_table.png

`make bench` invokes this module. `make bench-fast` sets BENCH_FAST=1 for the
smoke-test path (tiny dataset, 1 epoch).
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bench.data.synthetic_donations import generate_donations, split_chronological
from bench.data.synthetic_users import generate_users
from bench.eval import calibration, invariants
from bench.eval.metrics import (
    aggregate_metrics,
    catalog_coverage,
    mean_intra_list_diversity,
    user_coverage,
)
from bench.models.category_match import CategoryMatchRecommender
from bench.models.matrix_factorization import MatrixFactorizationRecommender
from bench.models.popularity import PopularityRecommender
from bench.models.random_baseline import RandomRecommender
from bench.models.two_tower import TwoTowerConfig, TwoTowerRecommender
from bench.models.two_tower_content_init import TwoTowerContentInitRecommender

logger = logging.getLogger(__name__)

BENCH_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = BENCH_ROOT / "results"
DATA_DIR = BENCH_ROOT / "data"

DEFAULT_SEED = 42

# Knobs for the headline run.
DEFAULT_N_USERS = 8000
DEFAULT_N_USERS_FAST = 200
DEFAULT_N_ORG_CAP = 3000  # sample down from the full ProPublica snapshot for speed
DEFAULT_N_ORG_CAP_FAST = 200


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BENCH_ROOT.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_orgs(fast: bool) -> pd.DataFrame:
    orgs_path = DATA_DIR / "orgs.csv"
    if not orgs_path.exists():
        raise FileNotFoundError(
            f"missing {orgs_path}. Run `python -m bench.scripts.fetch_propublica` first."
        )
    df = pd.read_csv(orgs_path, dtype=str).fillna("")
    cap = DEFAULT_N_ORG_CAP_FAST if fast else DEFAULT_N_ORG_CAP
    # Sample evenly per category for diversity.
    parts = []
    per_cat = max(20, cap // max(df["category"].nunique(), 1))
    for _, group in df.groupby("category"):
        parts.append(group.sample(min(len(group), per_cat), random_state=DEFAULT_SEED))
    sampled = pd.concat(parts, ignore_index=True)
    # Re-sample down to cap if we still overshot.
    if len(sampled) > cap:
        sampled = sampled.sample(n=cap, random_state=DEFAULT_SEED).reset_index(drop=True)
    return sampled


def build_dataset(*, seed: int, fast: bool) -> dict[str, Any]:
    orgs = _load_orgs(fast=fast)
    categories = sorted(orgs["category"].unique().tolist())
    n_users = DEFAULT_N_USERS_FAST if fast else DEFAULT_N_USERS
    users_df, profiles = generate_users(
        n_users=n_users, categories=categories, seed=seed
    )
    donations = generate_donations(
        user_profiles=profiles,
        orgs=orgs,
        seed=seed,
        donations_per_user_min=2 if fast else 3,
        donations_per_user_max=8 if fast else 25,
    )
    train, val, test = split_chronological(donations, val_frac=0.15, test_frac=0.2)
    return {
        "orgs": orgs,
        "users_df": users_df,
        "profiles": profiles,
        "donations": donations,
        "train": train,
        "val": val,
        "test": test,
        "categories": categories,
    }


def _build_user_test_truths(test: pd.DataFrame) -> dict[str, set[str]]:
    return {u: set(g["org_id"].tolist()) for u, g in test.groupby("user_id")}


def _eval_recommender(
    rec,
    *,
    test_users: list[str],
    truths: dict[str, set[str]],
    orgs: pd.DataFrame,
    k_top: int = 50,
    diversity_k: int = 10,
) -> dict[str, Any]:
    rankings = {u: rec.recommend(u, k_top) for u in test_users if u in truths}
    m = aggregate_metrics(rankings, truths, ks=[5, 10, 20, 50])
    item_cat = orgs.set_index("org_id")["category"].astype(str).to_dict()
    m["coverage@10"] = round(catalog_coverage(rankings, orgs["org_id"].tolist(), k=10), 4)
    m["user_coverage@10"] = round(user_coverage(rankings, test_users, k=10), 4)
    m["diversity@10"] = round(mean_intra_list_diversity(rankings, item_cat, k=diversity_k), 4)
    return {"rankings": rankings, "metrics": {k: round(float(v), 4) for k, v in m.items()}}


def _cold_user_split(
    truths: dict[str, set[str]],
    train: pd.DataFrame,
    threshold: int = 3,
) -> tuple[list[str], list[str]]:
    train_counts = train.groupby("user_id").size().to_dict() if not train.empty else {}
    cold = [u for u in truths if train_counts.get(u, 0) < threshold]
    warm = [u for u in truths if train_counts.get(u, 0) >= threshold]
    return cold, warm


def _cold_org_recall(
    rankings: dict[str, list[str]],
    truths: dict[str, set[str]],
    train: pd.DataFrame,
    threshold: int = 5,
    k: int = 10,
) -> float:
    org_counts = train.groupby("org_id").size().to_dict() if not train.empty else {}
    cold_orgs = {o for o, c in org_counts.items() if c < threshold}
    cold_orgs |= {o for o in set().union(*truths.values()) if o not in org_counts}
    if not cold_orgs:
        return 0.0
    hits = 0
    eligible = 0
    for user_id, truth in truths.items():
        cold_truth = truth & cold_orgs
        if not cold_truth:
            continue
        eligible += 1
        top_k = set(rankings.get(user_id, [])[:k])
        if top_k & cold_truth:
            hits += 1
    return hits / eligible if eligible else 0.0


def run_benchmark(*, fast: bool, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    t0 = time.time()
    bundle = build_dataset(seed=seed, fast=fast)
    orgs = bundle["orgs"]
    users_df = bundle["users_df"]
    profiles = bundle["profiles"]
    train = bundle["train"]
    val = bundle["val"]
    test = bundle["test"]
    truths = _build_user_test_truths(test)
    test_users = list(truths.keys())

    models = [
        RandomRecommender(seed=seed),
        PopularityRecommender(),
        CategoryMatchRecommender(),
        MatrixFactorizationRecommender(
            factors=24 if fast else 32,
            iterations=5 if fast else 15,
            seed=seed,
        ),
        TwoTowerRecommender(TwoTowerConfig(
            embed_dim=16 if fast else 32,
            epochs=1 if fast else 25,
            n_negatives=2 if fast else 5,
            fast_mode=fast,
            seed=seed,
        )),
        TwoTowerContentInitRecommender(TwoTowerConfig(
            embed_dim=16 if fast else 32,
            epochs=1 if fast else 15,
            n_negatives=2 if fast else 5,
            content_init=True,
            fast_mode=fast,
            seed=seed,
        )),
    ]

    results_per_model: dict[str, dict[str, Any]] = {}
    rankings_per_model: dict[str, dict[str, list[str]]] = {}
    centerpiece_ref: TwoTowerRecommender | None = None
    for model in models:
        logger.info("fitting %s ...", model.name)
        t1 = time.time()
        model.fit(train_donations=train, orgs=orgs, users=users_df)
        fit_seconds = time.time() - t1
        logger.info("evaluating %s ...", model.name)
        evald = _eval_recommender(
            model,
            test_users=test_users,
            truths=truths,
            orgs=orgs,
            k_top=50,
        )
        cold_users, warm_users = _cold_user_split(truths, train)
        if cold_users:
            cold_rankings = {u: evald["rankings"].get(u, []) for u in cold_users}
            cold_truths = {u: truths[u] for u in cold_users}
            cold_m = aggregate_metrics(cold_rankings, cold_truths, ks=[10])
            evald["metrics"]["cold_user_ndcg@10"] = round(cold_m.get("ndcg@10", 0.0), 4)
            evald["metrics"]["cold_user_recall@10"] = round(cold_m.get("recall@10", 0.0), 4)
        else:
            evald["metrics"]["cold_user_ndcg@10"] = None
            evald["metrics"]["cold_user_recall@10"] = None
        evald["metrics"]["cold_org_recall@10"] = round(
            _cold_org_recall(evald["rankings"], truths, train),
            4,
        )
        evald["metrics"]["fit_seconds"] = round(fit_seconds, 2)
        results_per_model[model.name] = evald["metrics"]
        rankings_per_model[model.name] = evald["rankings"]
        if isinstance(model, TwoTowerRecommender) and model.name == "two-tower":
            centerpiece_ref = model

    # Invariants run on the centerpiece (two-tower) — these gate the build.
    invariant_results = []
    if centerpiece_ref is not None:
        invariant_results.append(invariants.category_locked_invariant(
            centerpiece_ref, profiles, orgs, k=10,
            threshold=0.15 if fast else 0.4,  # fast mode trains 1 epoch — looser threshold
        ).to_dict())
        invariant_results.append(invariants.diversity_floor_invariant(
            centerpiece_ref, profiles, orgs, k=10,
        ).to_dict())
    invariant_results.append(invariants.beats_random_invariant(results_per_model).to_dict())
    all_invariants_pass = all(r["passed"] for r in invariant_results)

    # Calibration + training curve plots for the centerpiece.
    if centerpiece_ref is not None:
        cal = calibration.calibration_data(centerpiece_ref, test, n_users_sample=200)
        calibration.write_calibration_plot(cal, RESULTS_DIR / "calibration.png")
        _write_training_curve(centerpiece_ref, RESULTS_DIR / "training_curves.png")

    # Comparison table image.
    _write_comparison_image(results_per_model, RESULTS_DIR / "comparison_table.png")

    total_seconds = time.time() - t0

    metrics_summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "dataset_snapshot_date": _read_orgs_snapshot_date(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "seed": seed,
        "fast_mode": fast,
        "n_orgs": int(orgs.shape[0]),
        "n_users": int(users_df.shape[0]),
        "n_donations": int(bundle["donations"].shape[0]),
        "n_train": int(train.shape[0]),
        "n_val": int(val.shape[0]),
        "n_test": int(test.shape[0]),
        "runtime_seconds": round(total_seconds, 2),
        "models": results_per_model,
        "invariants": invariant_results,
        "invariants_all_pass": all_invariants_pass,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_summary, indent=2, default=str))

    # REPORT.md
    _write_report(metrics_summary, RESULTS_DIR / "REPORT.md")

    if not all_invariants_pass and not fast:
        # In fast mode we accept invariant violations because 1 epoch can't learn enough.
        logger.warning("invariants FAILED — see results/REPORT.md")
    return metrics_summary


def _read_orgs_snapshot_date() -> str:
    schema = DATA_DIR / "orgs_schema.md"
    if not schema.exists():
        return "unknown"
    try:
        for line in schema.read_text().splitlines():
            if line.lower().startswith("snapshot date"):
                return line.split("**")[1] if "**" in line else line
    except Exception:
        pass
    return "unknown"


def _write_training_curve(rec: TwoTowerRecommender, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in rec.training_log]
    losses = [entry["loss"] for entry in rec.training_log]
    fig, ax = plt.subplots(figsize=(6, 4))
    if epochs:
        ax.plot(epochs, losses, marker="o", color="#3366cc", label="train BPR loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax.set_title("Two-tower training curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_comparison_image(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(results.keys())
    metric_keys = ["ndcg@10", "recall@10", "mrr", "map@10", "coverage@10", "diversity@10"]
    n_models = len(model_names)
    n_metrics = len(metric_keys)
    arr = np.zeros((n_models, n_metrics))
    for i, name in enumerate(model_names):
        for j, key in enumerate(metric_keys):
            v = results[name].get(key)
            arr[i, j] = float(v) if v is not None else 0.0

    fig, ax = plt.subplots(figsize=(10, 0.6 * n_models + 2))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_keys, rotation=30, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names)
    for i in range(n_models):
        for j in range(n_metrics):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Donation Platform Recommender — Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: `{summary['generated_at']}`")
    lines.append(f"- Git SHA: `{summary['git_sha']}`")
    lines.append(f"- Org corpus snapshot: `{summary['dataset_snapshot_date']}`")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Runtime: `{summary['runtime_seconds']}s` on `{summary['platform']}` / Python `{summary['python_version']}`")
    lines.append(f"- Mode: `{'FAST (smoke)' if summary['fast_mode'] else 'full'}`")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Orgs: `{summary['n_orgs']:,}` (sampled across all NTEE major categories, sourced from ProPublica Nonprofit Explorer)")
    lines.append(f"- Synthetic users: `{summary['n_users']:,}`")
    lines.append(f"- Synthetic donation events: `{summary['n_donations']:,}` (train `{summary['n_train']:,}` / val `{summary['n_val']:,}` / test `{summary['n_test']:,}`)")
    lines.append("")
    lines.append("## Headline comparison")
    lines.append("")
    lines.append("Metrics are mean-over-users. Higher is better for ranking metrics; higher = more diverse for diversity@10.")
    lines.append("")
    metric_keys = ["ndcg@10", "recall@10", "mrr", "map@10", "coverage@10", "diversity@10", "cold_user_ndcg@10", "cold_user_recall@10", "cold_org_recall@10", "fit_seconds"]
    header = "| Model | " + " | ".join(metric_keys) + " |"
    sep = "|" + "|".join(["---"] * (len(metric_keys) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for name, row in summary["models"].items():
        cells = [name]
        for key in metric_keys:
            val = row.get(key)
            if val is None:
                cells.append("—")
            elif isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("![Comparison table](comparison_table.png)")
    lines.append("")
    lines.append("## Invariant tests")
    lines.append("")
    lines.append("| Invariant | Status | Score | Threshold | Notes |")
    lines.append("|---|---|---|---|---|")
    for inv in summary["invariants"]:
        status = "✅ PASS" if inv["passed"] else "❌ FAIL"
        lines.append(f"| {inv['name']} | {status} | {inv['score']:.4f} | {inv['threshold']:.4f} | {inv['notes']} |")
    lines.append("")
    lines.append("## Centerpiece plots")
    lines.append("")
    lines.append("- ![Training curve](training_curves.png)")
    lines.append("- ![Calibration](calibration.png)")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Run `make bench` from a clean checkout. Fixed seeds. Same git SHA = same numbers (up to floating-point determinism on CPU).")
    lines.append("- `make bench-fast` runs a tiny version in <30 seconds for CI smoke-testing.")
    lines.append("- `make bench-ablations` runs the 3×3×2 hyperparam sweep separately (~1 hr).")
    lines.append("")
    lines.append("See `bench/README.md` for the honesty footer and data provenance.")
    out_path.write_text("\n".join(lines))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fast = bool(os.environ.get("BENCH_FAST"))
    seed = int(os.environ.get("BENCH_SEED", DEFAULT_SEED))
    summary = run_benchmark(fast=fast, seed=seed)
    print(f"\nwrote {RESULTS_DIR / 'metrics.json'}")
    print(f"wrote {RESULTS_DIR / 'REPORT.md'}")
    if not summary["invariants_all_pass"] and not fast:
        print("WARNING: invariants did not all pass — see REPORT.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
