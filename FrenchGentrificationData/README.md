# French Gentrification Data — Paris Property Investment Signal

## What this is

An investment-signal map for Paris, built entirely from **legal, publicly
released French data** — property transaction prices, household income, and
occupation/education shift by neighborhood (IRIS zone). It deliberately does
**not** include race, ethnicity, or religion: French law prohibits collecting
or publishing that data (Article 8-1 of the Data Protection Act; see the
project discussion for sources). What's here instead is what real gentrification
economics is actually built from: **where are prices and incomes rising
together, and where has that trend not yet caught up to price.**

This is a data-exploration tool, not financial advice. It surfaces
historical, objective indicators — it does not predict future returns,
account for transaction costs/taxes, or replace due diligence on a specific
property.

## Why the files aren't here yet

This session's network access is blocked at the proxy level for
`data.gouv.fr`, `insee.fr`, `opendata.apur.org`, and `dvf.etalab.gouv.fr`
(confirmed: gateway returns 403 on CONNECT for all four). I can't download
these automatically. **Please download the files below and drop them into
the matching `raw/` subfolder** — once they're there, I'll build the
processing pipeline and the map.

## Files needed

### 1. `raw/dvf/` — property transaction prices (the core price signal)
Source: **DVF (Demandes de Valeurs Foncières)**, published by DGFiP via
data.gouv.fr / Etalab's "geo-dvf" project. Every real-estate transaction in
France — price, address, surface area, property type. No buyer/seller names.

- Page: https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres/
- Direct bulk files (geo-dvf, per year, per department), department 75 = Paris:
  `https://files.data.gouv.fr/geo-dvf/latest/csv/{YEAR}/departements/75.csv.gz`
- **Grab at least 3 years, spread out** so a price-trend can be computed —
  e.g. 2019, 2021, 2023 (or whatever the most recent available year is).
  Rename each to `dvf_75_{YEAR}.csv.gz` when you drop it in.

### 2. `raw/insee_income/` — household income by IRIS (Filosofi)
Source: **INSEE, "Revenus, pauvreté et niveau de vie" (Filosofi), IRIS level.**
Median income and poverty rate per IRIS zone — the standard input for "who
can afford to live here, and is that changing."

- Latest vintage (2021 incomes): https://www.insee.fr/fr/statistiques/8229323
- An earlier vintage for comparison (2017 incomes):
  https://www.insee.fr/fr/statistiques/4479212
- Download the IRIS-level file (usually `.xlsx` or `.csv`) for **at least
  two vintages** a few years apart — the gap between them is what lets us
  compute an income-growth trend, not just a snapshot.

### 3. `raw/insee_geometry/` — IRIS zone boundaries (needed to draw the map at all)
Source: **IGN / INSEE "Contours IRIS"** — the polygon boundaries for every
IRIS zone in Paris. Without this there's nothing to draw the overlay on.

- Look for "Contours IRIS" on data.gouv.fr (IGN publishes this; a GeoJSON
  or Shapefile for the current IRIS vintage). Filter/clip to Paris (75) if
  the file is national.

### 4. `raw/insee_occupation/` — occupation/education shift (optional but valuable)
Source: **INSEE census, IRIS-level "CSP" (socio-professional category) and
education-level tables.** The standard French gentrification proxy — rising
share of "cadres et professions intellectuelles supérieures" in a zone over
time is the textbook signal (this is literally how French urban sociologists,
e.g. Anne Clerval's work on Paris, measure gentrification).

- Search INSEE's IRIS data portal for "Population" / "Catégorie
  socioprofessionnelle" tables, again **two vintages** a few years apart.

### 5. `raw/apur/` — optional cross-check
Source: **APUR (Atelier Parisien d'Urbanisme)**, Paris's own planning
agency — opendata.apur.org. Has IRIS-level demographic and housing
indicators, and the official "quartiers prioritaires" (priority
neighbourhoods) boundaries, useful as a sanity check against whatever the
DVF/Filosofi trend shows.

## What happens once the files land

1. Clip everything to Paris intra-muros (the 20 arrondissements) by IRIS code.
2. Compute, per IRIS: price/m² trend (DVF), income trend (Filosofi), and
   occupation-shift trend (INSEE CSP) — each as a % change over the years
   you provide, not just a snapshot.
3. Combine into a simple, transparent "signal" score — e.g. rising
   income/occupation trend + price not yet fully caught up = higher signal;
   already-expensive + flat trend = lower signal. The exact formula will be
   shown, not hidden — you should be able to see and adjust the weights.
4. Render the low-alpha color overlay on the Paris IRIS map you described,
   plus the underlying data table.

## Folder structure

```
FrenchGentrificationData/
  raw/
    dvf/              <- geo-dvf CSVs, one per year
    insee_income/     <- Filosofi IRIS income files, 2+ vintages
    insee_geometry/   <- IRIS boundary GeoJSON/Shapefile
    insee_occupation/ <- INSEE CSP/education IRIS tables, 2+ vintages (optional)
    apur/             <- APUR IRIS indicators (optional)
  processed/           <- cleaned/joined output (generated, not manual)
```
