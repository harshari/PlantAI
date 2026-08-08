# US States Wish-List Analysis

An interactive 50-state screening for a relocation wish list, scored on seven
criteria and visualized three ways: a US choropleth map (red = unlikely fit,
green = strong fit), a ranked bar chart, and a taxes-vs-everything-else Pareto
scatter.

**Open `index.html` in any browser** — it is fully self-contained (no network,
no dependencies). It supports light/dark themes, per-criterion weight sliders,
hover breakdowns for every state, and a colorblind-safe palette toggle.

## Criteria (each scored 0–10 per state)

| Criterion | What it measures |
|---|---|
| Four-season climate | All four seasons present, summers not brutal (Austin-style two-season heat scores low) |
| Low taxes | Overall state + local tax burden (income, sales, property combined) |
| Reproductive rights | Abortion access and bodily-autonomy protections as of 2025 |
| Cannabis freedom | Recreational-legal scores high; used as a proxy for socially liberal policy |
| Healthcare & hospitals | Hospital quality, access, insurance coverage, outcomes |
| Police accountability | Low per-capita police-violence rates, oversight |
| Schools & universities | K-12 quality plus strength of higher-ed institutions |

The default composite is the equal-weighted mean; the sliders in the page
recompute it live with your own weights.

## Files

- `index.html` — the interactive report (map + bar chart + Pareto scatter + full table)
- `states_scores.csv` — the underlying scores, one row per state, with the equal-weighted composite

## Method and caveats

Scores are 0–10 judgment calls assembled from public state-level facts as of
2025 (climate normals, tax-burden rankings, post-*Dobbs* abortion law, cannabis
statutes, hospital/insurance metrics, police-violence rates, education
rankings). They are screening estimates meant for shortlisting, not precise
measurements — verify anything decisive for finalist states.

Statewide averages hide large in-state variation (California contains both
one-season San Diego and four-season Tahoe; Washington contains mild Seattle
and dry-winter Pullman). After shortlisting states, compare metro areas.

Map geometry: pre-projected Albers composite from the `us-atlas` package
(US Census Bureau cartographic boundaries), embedded as SVG paths.
