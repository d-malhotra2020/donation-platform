"""Recommender service — loads the trained two-tower + demo bundle on boot.

Designed to keep memory + boot time tight for Railway hobby tier:
  - No sentence-transformers at runtime (was a training-only dep)
  - No implicit at runtime
  - Just torch + numpy + faiss-cpu + pandas
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bench.models.two_tower import TwoTowerRecommender


class RecommenderService:
    """Loaded once at FastAPI startup. Thread-safe for read endpoints."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.model = TwoTowerRecommender.load_artifacts(artifacts_dir)
        self.orgs = pd.read_csv(artifacts_dir / "demo_orgs.csv", dtype=str).fillna("")
        self._orgs_by_id = self.orgs.set_index("org_id")
        self.users = pd.read_csv(artifacts_dir / "demo_users.csv").fillna("")
        self._users_by_id = self.users.set_index("user_id")
        self.donations = pd.read_csv(artifacts_dir / "demo_donations.csv")
        self.test_truth = pd.read_csv(artifacts_dir / "demo_test_truth.csv")
        self.metrics = json.loads((artifacts_dir / "demo_metrics.json").read_text())
        # Per-user training history for fast lookup.
        self._history_by_user: dict[str, pd.DataFrame] = {
            u: g.sort_values("timestamp_day") for u, g in self.donations.groupby("user_id")
        }
        self._truth_by_user: dict[str, set[str]] = {
            u: set(g["org_id"]) for u, g in self.test_truth.groupby("user_id")
        }

    # ---- public surfaces ------------------------------------------------

    def dataset_summary(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "n_orgs_total": m["n_orgs"],
            "n_users_total": m["n_users"],
            "n_donations_total": m["n_donations"],
            "n_orgs_in_demo": int(self.orgs.shape[0]),
            "n_users_in_demo": int(self.users.shape[0]),
            "n_donations_in_demo": int(self.donations.shape[0]),
            "dataset_snapshot_date": m["dataset_snapshot_date"],
            "git_sha": m["git_sha"],
            "bench_generated_at": m["generated_at"],
            "bench_seed": m["seed"],
            "bench_runtime_seconds": m["runtime_seconds"],
        }

    def comparison_table(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, row in self.metrics["models"].items():
            out.append({"model": name, **{k: row.get(k) for k in [
                "ndcg@10", "recall@10", "mrr", "map@10",
                "coverage@10", "diversity@10",
                "cold_user_ndcg@10", "cold_user_recall@10",
                "cold_org_recall@10", "fit_seconds",
            ]}})
        return out

    def invariants(self) -> list[dict[str, Any]]:
        return self.metrics["invariants"]

    def list_users(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = []
        # Sort: locked users first (most "demo-able"), then alphabetical.
        sorted_users = self.users.sort_values(
            by=["is_category_locked", "user_id"], ascending=[False, True]
        )
        for _, u in sorted_users.head(limit).iterrows():
            primary = str(u.get("primary_categories", ""))
            primary_list = [p for p in primary.split("|") if p]
            label = (
                f"locked → {primary_list[0]}" if u["is_category_locked"]
                else "multi-interest → " + " · ".join(primary_list[:3])
            )
            rows.append({
                "user_id": u["user_id"],
                "label": label,
                "is_category_locked": bool(u["is_category_locked"]),
                "primary_categories": primary_list,
            })
        return rows

    def user_detail(self, user_id: str, k: int = 10) -> dict[str, Any] | None:
        if user_id not in self._users_by_id.index:
            return None
        u = self._users_by_id.loc[user_id]
        history_df = self._history_by_user.get(user_id, pd.DataFrame())
        history_cats = Counter(history_df["category"].tolist()) if not history_df.empty else Counter()
        top_categories = [{"category": c, "count": n} for c, n in history_cats.most_common(5)]
        history_rows = []
        for _, row in history_df.tail(15).iterrows():
            org_id = row["org_id"]
            org = self._orgs_by_id.loc[org_id] if org_id in self._orgs_by_id.index else None
            history_rows.append({
                "org_id": org_id,
                "name": org["name"] if org is not None else org_id,
                "category": row["category"],
                "city": org["city"] if org is not None else "",
                "state": org["state"] if org is not None else "",
                "day": int(row["timestamp_day"]),
            })

        recs_raw = self.model.recommend(user_id, k)
        recs: list[dict[str, Any]] = []
        truth = self._truth_by_user.get(user_id, set())
        history_orgs = set(history_df["org_id"]) if not history_df.empty else set()
        top_history_cat = next(iter(history_cats), None) if history_cats else None
        for i, oid in enumerate(recs_raw):
            org = self._orgs_by_id.loc[oid] if oid in self._orgs_by_id.index else None
            if org is None:
                continue
            reasons = []
            if top_history_cat and org["category"] == top_history_cat:
                reasons.append(f"matches your top category ({top_history_cat})")
            elif history_cats.get(org["category"], 0) > 0:
                reasons.append(f"matches a secondary interest ({org['category']})")
            if oid in history_orgs:
                reasons.append("you've donated here before")
            if oid in truth:
                reasons.append("✓ in held-out test set")
            recs.append({
                "rank": i + 1,
                "org_id": oid,
                "name": org["name"],
                "category": org["category"],
                "city": org["city"],
                "state": org["state"],
                "in_truth": oid in truth,
                "reasons": reasons or ["learned latent affinity"],
            })

        return {
            "user_id": user_id,
            "is_category_locked": bool(u["is_category_locked"]),
            "primary_categories": [
                p for p in str(u.get("primary_categories", "")).split("|") if p
            ],
            "n_train_donations": int(len(history_df)),
            "history_top_categories": top_categories,
            "recent_history": list(reversed(history_rows)),
            "n_test_truth": len(truth),
            "test_truth_org_ids": sorted(truth),
            "recommendations": recs,
        }
