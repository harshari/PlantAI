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

## Second map: car-dwelling friendliness

The page also includes an independent choropleth scoring how each state treats
someone living out of their vehicle: greener = no statewide criminalization,
legal protections for vehicle homes (e.g. Washington's homestead-act case law),
safe-parking programs, tolerant rest areas/public land, and support services;
redder = statewide criminalization (Tennessee's 2022 felony public-camping law,
Texas's 2021 statewide ban, Florida's 2024 mandatory local bans, Kentucky's 2024
repeat-offense felony). Each state's tooltip carries a one-line rationale.
Day-to-day enforcement is mostly municipal — especially after *Grants Pass v.
Johnson* (2024) — so verify the specific city.

## Files

- `index.html` — the interactive report (two maps + bar chart + Pareto scatter + full table)
- `states_scores.csv` — the wish-list scores, one row per state, with the equal-weighted composite
- `car_dwelling_scores.csv` — the car-dwelling friendliness score and per-state rationale

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
