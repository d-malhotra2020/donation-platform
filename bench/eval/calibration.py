"""Calibration plot for the centerpiece model.

Bins recommended orgs by predicted score decile and plots the observed
positive rate per bin. A well-calibrated ranker is monotonic and close to y=x
(within rescaling). The plot tells the reader whether the model's scores are
*meaningful* numbers or just an arbitrary ranking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bench.models.two_tower import TwoTowerRecommender


def _user_scores(recommender: TwoTowerRecommender, user_id: str) -> tuple[list[str], np.ndarray]:
    """Return (org_ids, scores) for *all* orgs for the given user, sorted by score desc."""
    import torch

    if recommender.user_tower is None or recommender.org_embeddings is None:
        return [], np.array([])
    if user_id not in recommender.user_index:
        return [], np.array([])
    uidx = torch.tensor([recommender.user_index[user_id]], dtype=torch.long)
    with torch.no_grad():
        u_vec = recommender.user_tower(uidx).numpy().astype(np.float32)
    scores = recommender.org_embeddings @ u_vec[0]
    order = np.argsort(-scores)
    return [recommender.index_org[i] for i in order], scores[order]


def calibration_data(
    recommender: TwoTowerRecommender,
    test_donations: pd.DataFrame,
    n_users_sample: int = 200,
    n_bins: int = 10,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Produce (bin_centers, observed_rates, n_observations_per_bin) for plotting."""
    if test_donations.empty:
        return {"bin_centers": [], "observed_rates": [], "n_obs": []}

    rng = np.random.default_rng(seed)
    truth_per_user: dict[str, set[str]] = {
        u: set(g["org_id"].tolist())
        for u, g in test_donations.groupby("user_id")
    }
    candidate_users = [u for u in truth_per_user if u in recommender.user_index]
    if not candidate_users:
        return {"bin_centers": [], "observed_rates": [], "n_obs": []}
    sample = list(rng.choice(candidate_users, size=min(n_users_sample, len(candidate_users)), replace=False))

    all_scores: list[float] = []
    all_relevant: list[int] = []
    for user_id in sample:
        org_ids, scores = _user_scores(recommender, user_id)
        if not org_ids:
            continue
        relevant = truth_per_user[user_id]
        # Sample a fixed number of orgs per user to avoid dominating the global histogram.
        n_take = min(500, len(org_ids))
        idx = np.linspace(0, len(org_ids) - 1, n_take).astype(int)
        for i in idx:
            all_scores.append(float(scores[i]))
            all_relevant.append(1 if org_ids[i] in relevant else 0)

    if not all_scores:
        return {"bin_centers": [], "observed_rates": [], "n_obs": []}

    scores_arr = np.array(all_scores)
    relevant_arr = np.array(all_relevant)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(scores_arr, quantiles)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_ids = np.digitize(scores_arr, edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    centers: list[float] = []
    rates: list[float] = []
    counts: list[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        centers.append(float(scores_arr[mask].mean()))
        rates.append(float(relevant_arr[mask].mean()))
        counts.append(int(mask.sum()))
    return {"bin_centers": centers, "observed_rates": rates, "n_obs": counts}


def write_calibration_plot(
    data: dict[str, list[float]],
    out_path: Path,
    title: str = "Two-tower calibration (score decile vs observed positive rate)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if data["bin_centers"]:
        ax.plot(data["bin_centers"], data["observed_rates"], marker="o", color="#3366cc", label="observed")
        ax.set_xlabel("Predicted score (bin center)")
        ax.set_ylabel("Observed positive rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
