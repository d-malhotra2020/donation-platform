# `orgs.csv` schema

Snapshot date: **2026-05-24**
Row count: **5000**
Source: ProPublica Nonprofit Explorer (https://projects.propublica.org/nonprofits/api). Live API snapshot; only public search fields used.

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
| Arts, Culture & Humanities | 500 |
| Education | 500 |
| Environment & Animals | 500 |
| Health | 500 |
| Human Services | 500 |
| International | 500 |
| Public & Societal Benefit | 500 |
| Religion | 500 |
| Mutual & Membership Benefit | 500 |
| Other / Unknown | 500 |

## Refresh

Re-snapshot with `python -m bench.scripts.fetch_propublica`.
