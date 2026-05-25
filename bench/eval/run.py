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

    # Save the centerpiece model + a slim demo bundle for the live service.
    if centerpiece_ref is not None and not fast:
        artifacts_dir = BENCH_ROOT.parent / "app" / "artifacts"
        centerpiece_ref.save_artifacts(artifacts_dir)
        _write_demo_bundle(
            artifacts_dir,
            orgs=orgs,
            users_df=users_df,
            profiles=profiles,
            train=train,
            test=test,
            metrics=metrics_summary,
        )

    # REPORT.md + HTML landing pages
    _write_report(metrics_summary, RESULTS_DIR / "REPORT.md")
    _write_html_report(metrics_summary, RESULTS_DIR / "index.html", image_prefix="")
    docs_dir = BENCH_ROOT.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = "https://raw.githubusercontent.com/d-malhotra2020/donation-platform/master/bench/results/"
    _write_html_report(metrics_summary, docs_dir / "index.html", image_prefix=raw_prefix)

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


def _write_demo_bundle(
    out_dir: Path,
    *,
    orgs: pd.DataFrame,
    users_df: pd.DataFrame,
    profiles: list,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
) -> None:
    """Write a slim bundle the FastAPI service loads at boot.

    Only what the demo needs: orgs metadata, a sample of users with their
    train donation histories, and a frozen copy of the metrics summary for
    the static parts of the UI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    orgs[
        ["org_id", "name", "category", "ntee_major", "ntee_full", "city", "state"]
    ].to_csv(out_dir / "demo_orgs.csv", index=False)

    # Sample demo users with at least 3 train events, biased toward category-locked
    # so the demo UI has clean "this user only donates to environment, watch the
    # recommendations stay in category" stories.
    train_counts = train.groupby("user_id").size().to_dict()
    eligible = users_df[users_df["user_id"].map(train_counts).fillna(0) >= 3]
    locked_users = eligible[eligible["is_category_locked"] == True]
    multi_users = eligible[eligible["is_category_locked"] == False]
    n_locked = min(len(locked_users), 30)
    n_multi = min(len(multi_users), 170)
    demo_user_df = pd.concat([
        locked_users.sample(n=n_locked, random_state=0),
        multi_users.sample(n=n_multi, random_state=0),
    ]).reset_index(drop=True)
    demo_user_ids = set(demo_user_df["user_id"])

    demo_user_df.to_csv(out_dir / "demo_users.csv", index=False)
    train_for_demo = train[train["user_id"].isin(demo_user_ids)]
    train_for_demo[["user_id", "org_id", "category", "timestamp_day"]].to_csv(
        out_dir / "demo_donations.csv", index=False
    )
    test_for_demo = test[test["user_id"].isin(demo_user_ids)]
    test_for_demo[["user_id", "org_id", "category"]].to_csv(
        out_dir / "demo_test_truth.csv", index=False
    )

    import json
    (out_dir / "demo_metrics.json").write_text(json.dumps({
        "models": metrics["models"],
        "invariants": metrics["invariants"],
        "n_orgs": metrics["n_orgs"],
        "n_users": metrics["n_users"],
        "n_donations": metrics["n_donations"],
        "dataset_snapshot_date": metrics["dataset_snapshot_date"],
        "git_sha": metrics["git_sha"],
        "generated_at": metrics["generated_at"],
        "runtime_seconds": metrics["runtime_seconds"],
        "seed": metrics["seed"],
    }))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Donation Platform Recommender — Benchmark</title>
<meta name="description" content="Real-data benchmark of six recommender models on a ProPublica nonprofit corpus + synthetic giving patterns. Fully reproducible." />
<style>
  :root {{
    --bg: #0a0a0c;
    --bg-2: #14141a;
    --fg: #e8e8e8;
    --fg-dim: #9aa0a6;
    --accent: #7cf26b;
    --border: #2a2a30;
    --warn: #f2c94c;
    --bad: #f26b7c;
    --good: #7cf26b;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ background: var(--bg); color: var(--fg); margin: 0; padding: 0; }}
  body {{
    font: 14px/1.55 ui-sans-serif, system-ui, "Geist", "Inter", -apple-system, "Helvetica Neue", sans-serif;
    max-width: 980px;
    margin: 0 auto;
    padding: 48px 32px 64px;
  }}
  code, pre, .mono {{ font-family: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, monospace; }}
  h1 {{ font-size: 1.65rem; margin: 0 0 8px; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1.1rem; margin: 36px 0 10px; letter-spacing: -0.01em; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ font-size: 0.95rem; margin: 24px 0 6px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 0.04em; }}
  .lede {{ color: var(--fg-dim); margin-bottom: 24px; }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px 24px; padding: 12px 16px; border: 1px solid var(--border); border-radius: 6px; margin: 12px 0 24px; font-size: 12px; }}
  .meta div {{ display: flex; flex-direction: column; }}
  .meta .k {{ color: var(--fg-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .meta .v {{ font-family: "JetBrains Mono", monospace; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12.5px; }}
  th, td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: right; font-family: "JetBrains Mono", monospace; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: var(--bg-2); color: var(--fg-dim); font-weight: 500; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.05em; }}
  tbody tr.headline {{ background: rgba(124, 242, 107, 0.06); }}
  tbody tr.headline td:first-child::before {{ content: "▸ "; color: var(--accent); }}
  .invariants li {{ margin: 4px 0; }}
  .pass {{ color: var(--good); }}
  .fail {{ color: var(--bad); }}
  figure {{ margin: 16px 0; padding: 0; }}
  figure img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 4px; background: white; }}
  figcaption {{ color: var(--fg-dim); font-size: 12px; margin-top: 4px; }}
  .honesty {{ margin-top: 32px; padding: 16px 18px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-2); }}
  .honesty h3 {{ color: var(--warn); margin-top: 0; }}
  a {{ color: var(--accent); }}
  .footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--fg-dim); font-size: 12px; }}
  .copy {{ font-family: "JetBrains Mono", monospace; background: var(--bg-2); padding: 8px 12px; border-radius: 4px; border: 1px solid var(--border); display: inline-block; margin: 4px 0; }}
</style>
</head>
<body>

<h1>donation-platform · recommender benchmark</h1>
<p class="lede">Six models, real US-nonprofit corpus from ProPublica, synthetic giving patterns. <span class="mono">make bench</span> reproduces every number on this page in ~1.5 min on a CPU laptop.</p>

<div class="meta">
  <div><span class="k">generated</span><span class="v">{generated_at}</span></div>
  <div><span class="k">git sha</span><span class="v">{git_sha}</span></div>
  <div><span class="k">corpus snapshot</span><span class="v">{dataset_snapshot_date}</span></div>
  <div><span class="k">seed</span><span class="v">{seed}</span></div>
  <div><span class="k">runtime</span><span class="v">{runtime_seconds}s</span></div>
  <div><span class="k">orgs</span><span class="v">{n_orgs:,}</span></div>
  <div><span class="k">synthetic users</span><span class="v">{n_users:,}</span></div>
  <div><span class="k">events (train/val/test)</span><span class="v">{n_train:,} / {n_val:,} / {n_test:,}</span></div>
</div>

<h2>Headline comparison</h2>
<p>Metrics are mean-over-users. Higher is better for ranking metrics. <span class="mono">coverage@10</span> = fraction of org corpus appearing in any user's top-10; <span class="mono">diversity@10</span> = mean intra-list category entropy (higher = more diverse). The centerpiece <span class="mono">two-tower</span> row is highlighted.</p>
{comparison_table_html}

<figure>
  <img src="{image_prefix}comparison_table.png" alt="Comparison heatmap of all six models" />
  <figcaption>Heatmap of every metric × model. Generated by <span class="mono">bench/eval/run.py</span>.</figcaption>
</figure>

<h2>Invariant tests</h2>
<p>Pass/fail assertions modeled on the deep-dive's "synthetic users with known preference profiles" pattern. They gate the build — a real failure here means the recommender is doing the wrong thing on at least one well-defined user archetype.</p>
<ul class="invariants">
{invariants_html}
</ul>

<h2>Two-tower centerpiece</h2>
<p>PyTorch user-tower + org-tower with in-batch sampled-softmax + popularity-weighted negative sampling. L2-normalized outputs, FAISS exact inner-product for top-K retrieval. The <span class="mono">two-tower-content-init</span> ablation initializes the org tower from <span class="mono">sentence-transformers/all-MiniLM-L6-v2</span> embeddings before fine-tuning.</p>

<figure>
  <img src="{image_prefix}training_curves.png" alt="Two-tower BPR training loss per epoch" />
  <figcaption>Sampled-softmax loss per epoch.</figcaption>
</figure>

<figure>
  <img src="{image_prefix}calibration.png" alt="Two-tower calibration plot" />
  <figcaption>Predicted-score decile vs observed positive rate. A well-calibrated ranker is monotonic and close to a straight line.</figcaption>
</figure>

<h2>Reproduce locally</h2>
<div class="copy">git clone https://github.com/d-malhotra2020/donation-platform</div><br/>
<div class="copy">cd donation-platform && make bench</div>
<p>Fixed seeds, pinned dependencies (<span class="mono">bench/requirements.txt</span>), CPU-only by contract. Same git SHA → same metrics.json modulo runtime.</p>

<div class="honesty">
  <h3>What this is, and what it isn't</h3>
  <ul>
    <li>The <strong>org corpus</strong> is a sanitized snapshot of <a href="https://projects.propublica.org/nonprofits/api">ProPublica's Nonprofit Explorer</a> from {dataset_snapshot_date}. ~3K orgs sampled per-category from the full 5K snapshot for the headline run. CSV + fetch script live in the repo at <span class="mono">bench/data/</span>.</li>
    <li>The <strong>users and donations are synthetic</strong>, generated from the seeded process in <span class="mono">bench/data/synthetic_*.py</span>. Not real people, not real giving behavior.</li>
    <li>All eval metrics are computed on the synthetic test split. They measure model quality <em>on this synthetic giving pattern</em>. They do <strong>not</strong> represent performance on any real production donation platform.</li>
    <li>This is <strong>Slice 1</strong> of a four-slice rebuild. Slices 2–4 (FastAPI gateway, Redis embedding cache, web demo surface) are out of scope here and will land in follow-up cycles.</li>
  </ul>
</div>

<div class="footer">
  <a href="https://github.com/d-malhotra2020/donation-platform">github.com/d-malhotra2020/donation-platform</a> · <a href="https://drewmalhotra.com/work/donation-platform">deep-dive on drewmalhotra.com</a> · <a href="https://github.com/d-malhotra2020/donation-platform/blob/master/bench/results/REPORT.md">raw REPORT.md</a> · <a href="https://github.com/d-malhotra2020/donation-platform/blob/master/bench/results/metrics.json">metrics.json</a>
</div>

</body>
</html>
"""


def _write_html_report(summary: dict[str, Any], out_path: Path, *, image_prefix: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = ["ndcg@10", "recall@10", "mrr", "map@10", "coverage@10", "diversity@10", "cold_user_ndcg@10", "cold_org_recall@10"]
    rows: list[str] = []
    rows.append("<table><thead><tr><th>Model</th>" + "".join(f"<th>{k}</th>" for k in metric_keys) + "</tr></thead><tbody>")
    for name, row in summary["models"].items():
        highlight = ' class="headline"' if name == "two-tower" else ""
        cells = "".join(
            f"<td>{(f'{row.get(k):.4f}' if isinstance(row.get(k), float) else (row.get(k) if row.get(k) is not None else '—'))}</td>"
            for k in metric_keys
        )
        rows.append(f"<tr{highlight}><td>{name}</td>{cells}</tr>")
    rows.append("</tbody></table>")
    comparison_table_html = "\n".join(rows)

    inv_rows: list[str] = []
    for inv in summary["invariants"]:
        css = "pass" if inv["passed"] else "fail"
        status = "✅ PASS" if inv["passed"] else "❌ FAIL"
        inv_rows.append(
            f"<li><span class=\"{css}\">{status}</span> · <strong>{inv['name']}</strong> — "
            f"score <span class=\"mono\">{inv['score']:.4f}</span>, threshold <span class=\"mono\">{inv['threshold']:.4f}</span>. "
            f"<span class=\"mono\" style=\"color:var(--fg-dim)\">{inv['notes']}</span></li>"
        )
    invariants_html = "\n".join(inv_rows)

    html = HTML_TEMPLATE.format(
        generated_at=summary["generated_at"],
        git_sha=summary["git_sha"],
        dataset_snapshot_date=summary["dataset_snapshot_date"],
        seed=summary["seed"],
        runtime_seconds=summary["runtime_seconds"],
        n_orgs=summary["n_orgs"],
        n_users=summary["n_users"],
        n_train=summary["n_train"],
        n_val=summary["n_val"],
        n_test=summary["n_test"],
        comparison_table_html=comparison_table_html,
        invariants_html=invariants_html,
        image_prefix=image_prefix,
    )
    out_path.write_text(html)


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
