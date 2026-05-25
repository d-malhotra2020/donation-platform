"""Hyperparameter sweep for the centerpiece model.

`make bench-ablations` runs this separately from the main `make bench`. Sweeps:
  embed_dim       ∈ {16, 32, 64}
  n_negatives     ∈ {1, 5, 10}
  content_init    ∈ {False, True}

That's 18 training runs total, ~1 hr on CPU at the chosen dataset scale.
Output: ablations.json + ablations.png (heatmap-style table render).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench.eval.metrics import aggregate_metrics
from bench.eval.run import _build_user_test_truths, build_dataset
from bench.models.two_tower import TwoTowerConfig, TwoTowerRecommender


def run_ablations(
    *,
    out_dir: Path,
    seed: int = 42,
    fast: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(seed=seed, fast=fast)
    train = bundle["train"]
    test = bundle["test"]
    orgs = bundle["orgs"]
    users = bundle["users_df"]
    profiles = bundle["profiles"]
    test_users = sorted(set(test["user_id"]))[:1500]  # cap for sweep speed
    truths = _build_user_test_truths(test)

    grid = list(itertools.product([16, 32, 64], [1, 5, 10], [False, True]))
    results: list[dict] = []
    for embed_dim, n_neg, content_init in grid:
        cfg = TwoTowerConfig(
            embed_dim=embed_dim,
            n_negatives=n_neg,
            content_init=content_init,
            epochs=12 if not fast else 1,
            seed=seed,
        )
        rec = TwoTowerRecommender(cfg)
        rec.fit(train_donations=train, orgs=orgs, users=users)
        rankings = {u: rec.recommend(u, 10) for u in test_users if u in truths}
        m = aggregate_metrics(rankings, truths, ks=[10])
        results.append({
            "embed_dim": embed_dim,
            "n_negatives": n_neg,
            "content_init": content_init,
            "ndcg@10": round(m.get("ndcg@10", 0.0), 4),
            "recall@10": round(m.get("recall@10", 0.0), 4),
            "mrr": round(m.get("mrr", 0.0), 4),
        })

    out = {
        "results": results,
        "fast_mode": fast,
        "seed": seed,
    }
    json_path = out_dir / "ablations.json"
    json_path.write_text(json.dumps(out, indent=2))
    _render_ablation_plot(results, out_dir / "ablations.png")
    return out


def _render_ablation_plot(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    # Pivot: rows = embed_dim, cols = n_negatives, cells = ndcg@10. Separate plots per content_init.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, content_init in zip(axes, [False, True]):
        sub = df[df["content_init"] == content_init]
        pivot = sub.pivot(index="embed_dim", columns="n_negatives", values="ndcg@10")
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("n_negatives")
        ax.set_ylabel("embed_dim")
        ax.set_title(f"NDCG@10 (content_init={content_init})")
        for i, dim in enumerate(pivot.index):
            for j, neg in enumerate(pivot.columns):
                v = pivot.loc[dim, neg]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    import os
    from pathlib import Path as P

    fast = bool(os.environ.get("BENCH_FAST"))
    out_dir = P(__file__).resolve().parents[1] / "results"
    print("running ablation sweep...")
    out = run_ablations(out_dir=out_dir, fast=fast)
    print(f"wrote {out_dir / 'ablations.json'} ({len(out['results'])} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
