"""Two-tower variant with sentence-transformer org-tower initialization.

Thin wrapper that forces `content_init=True` in the config. Exists as a separate
module so the orchestrator can list it as a distinct row in the comparison
table without juggling config objects.
"""
from __future__ import annotations

from .two_tower import TwoTowerConfig, TwoTowerRecommender


class TwoTowerContentInitRecommender(TwoTowerRecommender):
    name = "two-tower-content-init"

    def __init__(self, config: TwoTowerConfig | None = None) -> None:
        cfg = config or TwoTowerConfig()
        cfg.content_init = True
        super().__init__(cfg)
