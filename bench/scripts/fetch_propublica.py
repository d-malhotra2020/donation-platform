"""Fetch a sample of US nonprofits from ProPublica Nonprofit Explorer.

One-shot tool. Writes bench/data/orgs.csv and bench/data/orgs_schema.md.
Not part of `make bench` — re-run manually to refresh the snapshot.

API docs: https://projects.propublica.org/nonprofits/api

ProPublica uses a condensed 10-category NTEE classification (`ntee[id]=1..10`).
Each ID maps to a broad category — we use ProPublica's labels verbatim so the
corpus is faithful to the source.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import date
from pathlib import Path
from urllib import error, request

API_BASE = "https://projects.propublica.org/nonprofits/api/v2/search.json"

# ProPublica's condensed NTEE major groups. ID -> (letter prefix, human label).
NTEE_GROUPS: list[tuple[int, str, str]] = [
    (1, "A", "Arts, Culture & Humanities"),
    (2, "B", "Education"),
    (3, "C", "Environment & Animals"),
    (4, "E", "Health"),
    (5, "K", "Human Services"),
    (6, "Q", "International"),
    (7, "S", "Public & Societal Benefit"),
    (8, "X", "Religion"),
    (9, "Y", "Mutual & Membership Benefit"),
    (10, "Z", "Other / Unknown"),
]

USER_AGENT = "donation-platform-bench/0.1 (educational benchmark)"


def fetch_category(ntee_id: int, label: str, max_orgs: int, attempts: int = 3) -> list[dict]:
    """Search ProPublica for orgs in a category. Paginate until we hit `max_orgs`."""
    orgs: list[dict] = []
    page = 0
    seen_eins: set[str] = set()
    while len(orgs) < max_orgs:
        url = f"{API_BASE}?ntee%5Bid%5D={ntee_id}&page={page}"
        body: dict | None = None
        for attempt in range(attempts):
            try:
                req = request.Request(url, headers={"User-Agent": USER_AGENT})
                with request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except (error.URLError, error.HTTPError, TimeoutError) as exc:
                if attempt == attempts - 1:
                    print(f"  page {page}: giving up ({exc})", file=sys.stderr)
                else:
                    time.sleep(1.5 * (attempt + 1))
        if body is None:
            break

        hits = body.get("organizations", []) or []
        if not hits:
            break
        added_this_page = 0
        for hit in hits:
            ein = str(hit.get("ein", ""))
            if not ein or ein in seen_eins:
                continue
            seen_eins.add(ein)
            ntee_code = hit.get("ntee_code", "") or hit.get("raw_ntee_code", "") or ""
            orgs.append({
                "ein": ein,
                "name": (hit.get("name") or "").strip(),
                "category": label,
                "ntee_major": ntee_code[:1] if ntee_code else "",
                "ntee_full": ntee_code,
                "city": (hit.get("city") or "").strip(),
                "state": (hit.get("state") or "").strip(),
            })
            added_this_page += 1
            if len(orgs) >= max_orgs:
                break

        page += 1
        time.sleep(0.4)  # polite pacing
        if added_this_page == 0:
            break
        # Safety: ProPublica caps total_results at 10000 across pages.
        if page > 400:
            break
    return orgs[:max_orgs]


def fetch_all(per_category: int) -> list[dict]:
    out: list[dict] = []
    for ntee_id, letter, label in NTEE_GROUPS:
        print(f"[{ntee_id}/10] fetching {label}...", flush=True)
        cat_orgs = fetch_category(ntee_id, label, per_category)
        print(f"  got {len(cat_orgs)} orgs", flush=True)
        out.extend(cat_orgs)
    return out


def synthesize_fallback(per_category: int, seed: int = 42) -> list[dict]:
    """Generate a synthetic-but-plausible corpus when the API is unavailable."""
    rng = random.Random(seed)
    name_prefixes = [
        "Foundation", "Coalition", "Alliance", "Society", "Initiative",
        "Trust", "Network", "Project", "Center", "Institute", "Council",
        "Partnership", "Fund", "Association", "Collective",
    ]
    cities = [
        ("San Francisco", "CA"), ("New York", "NY"), ("Austin", "TX"),
        ("Chicago", "IL"), ("Seattle", "WA"), ("Denver", "CO"),
        ("Boston", "MA"), ("Atlanta", "GA"), ("Portland", "OR"),
        ("Minneapolis", "MN"), ("Nashville", "TN"), ("Phoenix", "AZ"),
    ]
    out: list[dict] = []
    for cat_idx, (_, letter, label) in enumerate(NTEE_GROUPS):
        for i in range(per_category):
            prefix = name_prefixes[(cat_idx * 7 + i) % len(name_prefixes)]
            city, state = cities[(cat_idx * 11 + i * 3) % len(cities)]
            # 99-prefixed pseudo-EIN to make it obviously non-real (real EINs are 9 digits).
            ein = f"99{cat_idx:02d}{i:05d}"
            short = label.split(",")[0].split("&")[0].strip()
            name = f"{short} {prefix} #{i + 1:04d}"
            out.append({
                "ein": ein,
                "name": name,
                "category": label,
                "ntee_major": letter,
                "ntee_full": f"{letter}01",
                "city": city,
                "state": state,
            })
    rng.shuffle(out)
    return out


def write_outputs(orgs: list[dict], out_dir: Path, source_note: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "orgs.csv"
    schema_path = out_dir / "orgs_schema.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "org_id", "ein", "name", "category", "ntee_major", "ntee_full",
                "city", "state",
            ],
        )
        writer.writeheader()
        for idx, org in enumerate(orgs):
            writer.writerow({"org_id": f"org_{idx:06d}", **org})

    from collections import Counter
    cat_counts = Counter(o["category"] for o in orgs)
    cat_breakdown = "\n".join(
        f"| {label} | {cat_counts.get(label, 0)} |"
        for _, _, label in NTEE_GROUPS
    )

    schema_path.write_text(
        f"""# `orgs.csv` schema

Snapshot date: **{date.today().isoformat()}**
Row count: **{len(orgs)}**
Source: {source_note}

## Columns

| Column        | Type   | Description |
|---------------|--------|-------------|
| `org_id`      | string | Stable internal id (`org_NNNNNN`). |
| `ein`         | string | IRS Employer Identification Number when available; synthetic-fallback rows use a `99`-prefixed non-IRS pattern. |
| `name`        | string | Organization name. |
| `category`    | string | ProPublica's NTEE major category label. |
| `ntee_major`  | string | NTEE major group letter (A–Y, Z=unknown). |
| `ntee_full`   | string | Full NTEE code (e.g. `A50Z`) when available. |
| `city`        | string | City. |
| `state`       | string | US state code. |

## Category breakdown

| Category | Count |
|----------|-------|
{cat_breakdown}

## Refresh

Re-snapshot with `python -m bench.scripts.fetch_propublica`.
""",
        encoding="utf-8",
    )
    print(f"\nwrote {csv_path} ({len(orgs)} rows)")
    print(f"wrote {schema_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a US nonprofit corpus snapshot.")
    parser.add_argument("--per-category", type=int, default=500, help="Max orgs per NTEE major category.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Output directory (defaults to bench/data).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the API entirely; produce a synthetic-fallback corpus.",
    )
    args = parser.parse_args()

    orgs: list[dict] = []
    used_fallback = False
    if not args.offline:
        try:
            orgs = fetch_all(args.per_category)
        except Exception as exc:  # noqa: BLE001
            print(f"API fetch failed entirely: {exc}", file=sys.stderr)
            orgs = []
        # Require at least 80% of target per category on average — otherwise fall back.
        expected = args.per_category * len(NTEE_GROUPS) * 0.8
        if len(orgs) < expected:
            print(
                f"only {len(orgs)} orgs from API (wanted >= {int(expected)}); using synthetic fallback",
                file=sys.stderr,
            )
            orgs = []

    if not orgs:
        used_fallback = True
        orgs = synthesize_fallback(args.per_category)

    source_note = (
        "Synthetic fallback corpus generated by `bench/scripts/fetch_propublica.py --offline`. "
        "Org names, descriptions, and EINs are synthetic (EINs prefixed with `99` to make this obvious). "
        "Category labels are real NTEE major groups; city/state are real US locations."
        if used_fallback
        else "ProPublica Nonprofit Explorer (https://projects.propublica.org/nonprofits/api). "
        "Live API snapshot; only public search fields used."
    )
    write_outputs(orgs, args.out, source_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
