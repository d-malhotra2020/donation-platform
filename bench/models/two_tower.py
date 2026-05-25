"""Two-tower neural recommender (PyTorch).

Centerpiece model. User tower + org tower → L2-normalized embeddings in shared
space. Trained on (user, donated_org) positives + popularity-weighted negatives
with BPR loss. Top-K served via FAISS exact inner-product (5K orgs is small
enough to skip ANN).

The org tower can optionally be initialized from sentence-transformer
embeddings — the `content_init` flag swaps in pre-trained vectors that the
fine-tuning then nudges. The content-init variant tends to help most on
cold-start orgs (the lesson the deep-dive page already calls out).

This module is the only place in `bench/` that imports torch. Everything
downstream of it just consumes the `recommend()` interface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Recommender

logger = logging.getLogger(__name__)


@dataclass
class TwoTowerConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.05
    n_negatives: int = 8
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    early_stop_patience: int = 5
    seed: int = 42
    content_init: bool = False
    content_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fast_mode: bool = False  # smoke-test path: tiny dataset, 1 epoch
    use_in_batch_negatives: bool = True  # standard two-tower training trick
    temperature: float = 0.07


class _UserTower(nn.Module):
    def __init__(self, n_users: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_users, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.normal_(self.embed.weight, std=0.05)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(user_idx)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class _OrgTower(nn.Module):
    def __init__(
        self,
        n_orgs: int,
        n_categories: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        content_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_orgs, embed_dim)
        self.cat_embed = nn.Embedding(n_categories, embed_dim)
        self.has_content = content_dim > 0
        input_dim = embed_dim * 2 + (content_dim if self.has_content else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.normal_(self.cat_embed.weight, std=0.05)

    def set_content(self, content: torch.Tensor) -> None:
        self.register_buffer("content_buf", content, persistent=False)

    def forward(self, org_idx: torch.Tensor, cat_idx: torch.Tensor) -> torch.Tensor:
        parts = [self.embed(org_idx), self.cat_embed(cat_idx)]
        if self.has_content:
            parts.append(self.content_buf[org_idx])
        x = torch.cat(parts, dim=-1)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerRecommender(Recommender):
    name = "two-tower"

    def __init__(self, config: TwoTowerConfig | None = None) -> None:
        self.cfg = config or TwoTowerConfig()
        if self.cfg.content_init:
            self.name = "two-tower-content-init"
        self.user_index: dict[str, int] = {}
        self.org_index: dict[str, int] = {}
        self.cat_index: dict[str, int] = {}
        self.index_org: list[str] = []
        self.org_cat_idx: torch.Tensor | None = None
        self.user_tower: _UserTower | None = None
        self.org_tower: _OrgTower | None = None
        self.org_embeddings: np.ndarray | None = None  # precomputed at fit-time
        self.popularity_fallback: list[str] = []
        self.faiss_index: Any | None = None
        # Optional: per-epoch metrics used by run.py for training-curve plots.
        self.training_log: list[dict] = field(default_factory=list)  # type: ignore[assignment]
        self.training_log = []

    # ---- fit ------------------------------------------------------------

    def fit(self, *, train_donations: pd.DataFrame, orgs: pd.DataFrame, users: pd.DataFrame) -> None:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        all_users = users["user_id"].tolist()
        all_orgs = orgs["org_id"].tolist()
        all_cats = orgs["category"].astype(str).unique().tolist()
        self.user_index = {u: i for i, u in enumerate(all_users)}
        self.org_index = {o: i for i, o in enumerate(all_orgs)}
        self.cat_index = {c: i for i, c in enumerate(all_cats)}
        self.index_org = all_orgs
        org_cat = orgs.set_index("org_id")["category"].astype(str).to_dict()
        self.org_cat_idx = torch.tensor(
            [self.cat_index[org_cat[o]] for o in all_orgs], dtype=torch.long
        )

        # Popularity fallback for cold-start users.
        if not train_donations.empty:
            pop = train_donations["org_id"].value_counts()
            seen = set(pop.index.tolist())
            self.popularity_fallback = pop.index.tolist() + [o for o in all_orgs if o not in seen]
            pop_counts = pop.reindex(all_orgs, fill_value=0).to_numpy(dtype=np.float64)
        else:
            self.popularity_fallback = all_orgs
            pop_counts = np.ones(len(all_orgs), dtype=np.float64)

        # Build positive pairs (filter to known users + orgs).
        pos_df = train_donations[
            train_donations["user_id"].isin(self.user_index)
            & train_donations["org_id"].isin(self.org_index)
        ]
        if pos_df.empty:
            logger.warning("two-tower: no training pairs — falling back to popularity-only inference")
            self._build_inference_artifacts(content=None)
            return

        u_idx = torch.tensor(pos_df["user_id"].map(self.user_index).to_numpy(), dtype=torch.long)
        o_idx = torch.tensor(pos_df["org_id"].map(self.org_index).to_numpy(), dtype=torch.long)

        # Content init: optionally pull pre-trained org descriptions into the tower.
        content_tensor: torch.Tensor | None = None
        content_dim = 0
        if self.cfg.content_init:
            content_tensor = self._compute_content_embeddings(orgs)
            content_dim = content_tensor.shape[1]

        self.user_tower = _UserTower(
            n_users=len(all_users),
            embed_dim=self.cfg.embed_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        )
        self.org_tower = _OrgTower(
            n_orgs=len(all_orgs),
            n_categories=len(all_cats),
            embed_dim=self.cfg.embed_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            content_dim=content_dim,
        )
        if content_tensor is not None:
            self.org_tower.set_content(content_tensor)

        optimizer = torch.optim.Adam(
            list(self.user_tower.parameters()) + list(self.org_tower.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # Popularity-weighted negative sampling distribution.
        neg_probs = pop_counts ** 0.75
        neg_probs = neg_probs / neg_probs.sum()
        neg_probs_t = torch.tensor(neg_probs, dtype=torch.double)

        n_pos = len(u_idx)
        epochs = 1 if self.cfg.fast_mode else self.cfg.epochs
        batch_size = min(self.cfg.batch_size, max(64, n_pos))
        best_loss = float("inf")
        patience_left = self.cfg.early_stop_patience

        for epoch in range(epochs):
            self.user_tower.train()
            self.org_tower.train()
            perm = torch.randperm(n_pos, generator=torch.Generator().manual_seed(self.cfg.seed + epoch))
            losses: list[float] = []
            for start in range(0, n_pos, batch_size):
                end = min(start + batch_size, n_pos)
                batch_idx = perm[start:end]
                u = u_idx[batch_idx]
                pos_o = o_idx[batch_idx]
                B = len(u)

                u_vec = self.user_tower(u)  # (B, D)
                pos_vec = self.org_tower(pos_o, self.org_cat_idx[pos_o])  # (B, D)

                # Sampled negatives (popularity-weighted) for hard signal.
                neg_o = torch.multinomial(neg_probs_t, num_samples=B * self.cfg.n_negatives, replacement=True).long()
                neg_vec = self.org_tower(neg_o, self.org_cat_idx[neg_o])  # (B*N, D)

                if self.cfg.use_in_batch_negatives:
                    # In-batch sampled-softmax: for each user in the batch, treat the
                    # other positives in the batch + the sampled negatives as
                    # candidates. The positive's logit must beat all of them.
                    all_neg = torch.cat([pos_vec, neg_vec], dim=0)  # (B + B*N, D)
                    pos_score = (u_vec * pos_vec).sum(dim=-1)  # (B,)
                    all_score = u_vec @ all_neg.T  # (B, B+B*N)
                    # Mask the diagonal — each user's own positive is the gold class.
                    diag_mask = torch.zeros_like(all_score, dtype=torch.bool)
                    diag_mask[torch.arange(B), torch.arange(B)] = True
                    # Logits: replace diagonal entries with pos_score (already there).
                    # Apply temperature.
                    logits = all_score / self.cfg.temperature
                    target = torch.arange(B)
                    loss = F.cross_entropy(logits, target)
                else:
                    neg_vec_r = neg_vec.view(B, self.cfg.n_negatives, -1)
                    pos_score = (u_vec * pos_vec).sum(dim=-1, keepdim=True)  # (B, 1)
                    neg_score = torch.einsum("bd,bnd->bn", u_vec, neg_vec_r)  # (B, N)
                    loss = -F.logsigmoid(pos_score - neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))

            epoch_loss = float(np.mean(losses)) if losses else 0.0
            self.training_log.append({"epoch": epoch, "loss": epoch_loss})
            logger.info("two-tower epoch %d/%d loss=%.4f", epoch + 1, epochs, epoch_loss)

            if epoch_loss + 1e-5 < best_loss:
                best_loss = epoch_loss
                patience_left = self.cfg.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0 and not self.cfg.fast_mode:
                    logger.info("two-tower early-stop at epoch %d", epoch + 1)
                    break

        self._build_inference_artifacts(content=content_tensor)

    # ---- inference ------------------------------------------------------

    def _build_inference_artifacts(self, content: torch.Tensor | None) -> None:
        if self.user_tower is None or self.org_tower is None or self.org_cat_idx is None:
            # No training happened — fall back to popularity at recommend time.
            self.org_embeddings = None
            self.faiss_index = None
            return

        self.user_tower.eval()
        self.org_tower.eval()
        with torch.no_grad():
            org_idx = torch.arange(len(self.index_org), dtype=torch.long)
            org_emb = self.org_tower(org_idx, self.org_cat_idx).numpy().astype(np.float32)
        self.org_embeddings = org_emb

        # FAISS index for top-K (inner-product on L2-normalized vectors == cosine).
        try:
            import faiss  # noqa: WPS433

            index = faiss.IndexFlatIP(org_emb.shape[1])
            index.add(org_emb)
            self.faiss_index = index
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable (%s) — using numpy top-K fallback", exc)
            self.faiss_index = None

    def recommend(self, user_id: str, k: int) -> list[str]:
        if self.user_tower is None or self.org_embeddings is None or user_id not in self.user_index:
            return self.popularity_fallback[:k]
        uidx = torch.tensor([self.user_index[user_id]], dtype=torch.long)
        with torch.no_grad():
            u_vec = self.user_tower(uidx).numpy().astype(np.float32)
        if self.faiss_index is not None:
            _, ids = self.faiss_index.search(u_vec, k)
            return [self.index_org[i] for i in ids[0] if i >= 0]
        # Numpy fallback.
        scores = self.org_embeddings @ u_vec[0]
        idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.index_org[i] for i in idx]

    # ---- content init ---------------------------------------------------

    def _compute_content_embeddings(self, orgs: pd.DataFrame) -> torch.Tensor:
        """Encode org name+category+state with a small sentence transformer.

        Cached on disk under bench/data/content_embeddings.npy keyed by org count
        + model name + a content hash so reruns don't redownload or recompute.
        """
        from pathlib import Path
        import hashlib

        descriptions = [
            f"{row['name']} | {row['category']} | {row.get('city', '')}, {row.get('state', '')}"
            for _, row in orgs.iterrows()
        ]
        cache_dir = Path(__file__).resolve().parents[1] / "data"
        sig = hashlib.sha1(("\n".join(descriptions) + self.cfg.content_model_name).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"content_embeddings_{sig}.npy"
        if cache_path.exists():
            arr = np.load(cache_path)
            logger.info("loaded cached content embeddings from %s", cache_path)
            return torch.tensor(arr, dtype=torch.float32)

        from sentence_transformers import SentenceTransformer

        logger.info("encoding %d org descriptions with %s ...", len(descriptions), self.cfg.content_model_name)
        model = SentenceTransformer(self.cfg.content_model_name)
        embeddings = model.encode(
            descriptions,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        arr = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_path, arr)
        return torch.tensor(arr, dtype=torch.float32)
