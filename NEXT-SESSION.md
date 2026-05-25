# Next-session entry points — donation-platform

**Session of 2026-05-24/25 (continuous, ~5 hr) closed with:** Slice 1 (bench/) live, all 3 presentation tiers shipped (GitHub Pages static report, raw-PNG inline plots on portfolio deep-dive, live operator console on Railway). See `SPEC.md` for the original design contract and `bench/README.md` for the honesty footer / reproducibility contract.

**Currently live:**

| Surface | URL |
|---|---|
| Operator console (FastAPI + trained two-tower) | https://donation-platform-production-c8e0.up.railway.app |
| Static benchmark report (GitHub Pages, from `/docs`) | https://d-malhotra2020.github.io/donation-platform/ |
| Portfolio deep-dive (real-org inline plots) | https://drewmalhotra.com/work/donation-platform |
| Portfolio card #003 | https://drewmalhotra.com (Projects.jsx) |
| Status board ping target | StatusBoard.jsx id `donate` |

**Headline numbers (last `make bench`, git SHA `ae4fb36`, dataset snapshot 2026-05-24):**

| Model | NDCG@10 | Catalog coverage@10 |
|---|---|---|
| random | 0.0021 | 100% |
| popularity | 0.0064 | 0.33% |
| category-match | **0.0255** | 3.33% |
| matrix-factorization | 0.0212 | 14.93% |
| two-tower | 0.0120 (5.7× rand, 1.9× pop) | **99.13%** |
| two-tower-content-init | 0.0109 | 97.30% |

Invariants: ✅ category-locked, ✅ beats-random, ❌ diversity-floor (real failure mode — see follow-up #2 below).

---

## Menu of next moves, ranked by ROI

### If you have ~1 hour

**#2 — Fix the diversity-floor invariant via MMR re-ranking** *(highest leverage for time)*. The two-tower sometimes returns top-10 lists that are 100% one category for multi-interest users. Standard fix is Maximal Marginal Relevance at inference: select rec_1 by score, then each subsequent rec by `λ·score - (1-λ)·max_similarity_to_already_chosen`. Implementation: add an MMR pass to `RecommenderService.user_detail` (or `TwoTowerRecommender.recommend`) with `λ=0.7`. Re-run `make bench`. Invariant should flip green. Update REPORT.md narrative. Commit + push (auto-deploys to Railway).

### If you have a half-day

**#1 — Slice 2: the gateway with fallback.** Wrap the trained two-tower with a thin FastAPI gateway that adds:
- Lexical-search fallback (BM25 over org names/descriptions) when the recommender is slow/down
- Feature-flag off-switch (`RECOMMENDER_ENABLED=false` → always lexical)
- Cold-start handling: explicit "popularity overlay" for users with <3 train donations
- Timeouts (e.g. 200ms budget per request before fall-through)

This is the operational story the deep-dive narrates but only Slice 1 exists today. After Slice 2, the deep-dive can say "and here's the off-switch I learned to build in v2."

**#3 — Run `make bench-ablations`.** The 3×3×2 sweep is built but never executed. ~1 hr CPU run. Produces an ablation heatmap that goes on the GitHub Pages report and answers the "did you pick the right hyperparameters?" question with data.

### If you want a more compelling headline

**#4 — Better synthetic data → two-tower wins NDCG@10.** Current dataset is category-dominated by construction so category-match wins on the headline metric. The two-tower wins on coverage by 30× but loses on NDCG@10. Honest, but a more compelling narrative is "two-tower wins on both" — achievable by adding temporal effects to `bench/data/synthetic_donations.py`:
- Popularity decay: hot orgs cool over time
- User lifecycle: new users explore broadly then narrow
- Time-of-day / day-of-week effects (give the model a feature category-match can't access)

Open-source data generator means this stays honest as long as the changes are documented.

### If the goal is more polish, not more scope

**#5 — Real model card.** `app/MODEL_CARD.md` in the Hugging Face / Google style — intended use, limitations, training data, evaluation, ethical considerations. Signals ML-ops thinking.

**Operator-console UX improvements:**
- "Diff against popularity baseline" toggle — show how the two-tower's top-10 differs from what popularity would have shown
- Org search/filter (not just user picker)
- Side-by-side: pick two users, compare their recs
- Click a recommendation → expanded "why" panel
- Screencast / animated GIF on the deep-dive page

**Code structure:**
- `bench/models/two_tower.py` is ~430 LOC — split into `model.py` (towers) + `trainer.py` (BPR loop + save/load)
- GitHub Actions: `make bench-fast` smoke test on every push (already smoke-testable in ~45s)
- pytest coverage badge for `bench/eval/metrics.py`

---

## Side-project pass (broader scope, separate cycle)

Per portfolio STATE.md, the side-project polish queue continues:
- **video-analytics** — next in the playbook queue. Same arc: replace fabricated numbers, build a real benchmark, ship a measured demo. Card #001 currently claims "500+ streams · 4,600+ alerts · 92% accuracy" — unverified.
- **traffic-optimization deep-dive page** — `src/work/` has 4 of 6 projects. Traffic-optimization isn't one of them despite being the most recently shipped real artifact.
- **qa-webhook-server** — orphaned repo at `~/projects/`, no remote, decision needed on whether to surface or retire.

---

## Drew-actions still open (consolidated, 2026-05-25)

1. **Anthropic dashboard:** $10/mo cap + email alerts at $5 / $9 (portfolio Phase 10 SC, still open)
2. **Google Search Console:** submit `https://drewmalhotra.com/sitemap.xml` (portfolio Phase 5)
3. **Cloudflare Analytics Engine:** enable in dashboard, uncomment `[[analytics_engine_datasets]]` in `workers/agent/wrangler.toml`, deploy — unblocks portfolio Phase 8
4. **LinkedIn recs:** 2–3 from ex-Brivo / Yunex / Givelify colleagues — unblocks portfolio Phase 6
5. **Mobile UX deep-pass on real iOS + Android** — unblocks portfolio Phase 11
6. **smart-home Mosquitto sidecar** — add as a Railway service, set env vars on the Flask service
7. **Diverged repos:** `~/projects/<project>/` April Railway variants still un-pushed

---

## How to resume

```bash
# Reproduce the bench locally (validates the artifacts haven't drifted):
cd ~/separate-projects/donation-platform
make bench-fast   # ~45s smoke test

# Verify the live deploy is still up:
curl -s https://donation-platform-production-c8e0.up.railway.app/api/v1/health

# Then pick a move from the menu above. Recommended starter: #2 (MMR diversity fix).
```

Bench venv lives at `.venv-bench/` (pinned deps via `bench/requirements.txt`). The trained model + demo bundle is at `app/artifacts/` (committed; ~2.6 MB total).
